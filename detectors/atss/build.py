"""
ATSS builder (Image task only).

Paper: Zhang, Chi, Yao, Lei, Li, "Bridging the Gap Between Anchor-based and Anchor-free Detection via Adaptive
       Training Sample Selection" (CVPR 2020, arXiv 1912.02424).
Code : official sfzhang15/ATSS (maskrcnn-benchmark based), configs/atss/atss_R_50_FPN_1x.yaml:
       atss_core/modeling/rpn/atss/atss.py   (BoxCoder 'BOX', ATSSHead)
       atss_core/modeling/rpn/atss/loss.py   (ATSSLossComputation, POSITIVE_TYPE 'ATSS', TOPK 9, GIoU x 2.0)
       atss_core/modeling/rpn/atss/inference.py (score = sqrt(sigmoid(cls) * sigmoid(centerness)))
       atss_core/modeling/rpn/anchor_generator.py (generate_anchors, make_anchor_generator_atss: size 8 x stride)
       Backbone / FPN: shared with detectors/retinanet (ResNet-50, FPN P3-P7 with P6/P7 from P5: USE_C5 False).
       maskrcnn-benchmark plumbing (BoxList, distributed reduce_sum, CUDA SigmoidFocalLoss) replaced by plain
       tensors / the equivalent fvcore focal loss; np.float -> np.float64 (numpy 2).

    one anchor per location (8S, ratio 1), ATSS positive selection: per GT the 9 closest anchors of every level,
    IoU threshold = mean + std of their IoUs, anchor centre inside the GT; ties -> highest IoU.
    loss : focal / #pos + 2.0 x centerness-weighted GIoU / sum(centerness) + BCE centerness / #pos

[Unified] as RetinaNet / FCOS: no ImageNet weights, trainable BN, common init (conv weights only), no mean / std,
608 input, Adam 1e-4, common post-processing (instead of score 0.05, top-1000, NMS 0.6), PreciseBN.
"""


import torch
import torch.nn as nn

from detectors.retinanet.box_ops import pairwise_iou
from detectors.retinanet.loss import sigmoid_focal_loss

INF = 100000000
TOPK = 9
REG_LOSS_WEIGHT = 2.0


# ---- atss_core/modeling/rpn/anchor_generator.py --------------------------------------------------------------


# ---- atss_core/modeling/rpn/atss/atss.py ---------------------------------------------------------------------


class Scale(nn.Module):
    """atss_core/layers/scale.py"""

    def __init__(self, init_value=1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


# ---- atss_core/modeling/rpn/atss/loss.py ---------------------------------------------------------------------
class ATSSLoss:
    def __init__(self, num_classes, box_coder, topk=TOPK, num_anchors_per_loc=1):
        self.num_classes = num_classes
        self.box_coder = box_coder
        self.topk = topk
        self.num_anchors_per_loc = num_anchors_per_loc

    def GIoULoss(self, pred, target, anchor, weight=None):
        pred_boxes = self.box_coder.decode(pred.view(-1, 4), anchor.view(-1, 4))
        pred_x1, pred_y1, pred_x2, pred_y2 = pred_boxes.unbind(1)
        pred_x2 = torch.max(pred_x1, pred_x2)
        pred_y2 = torch.max(pred_y1, pred_y2)
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        gt_boxes = self.box_coder.decode(target.view(-1, 4), anchor.view(-1, 4))
        target_x1, target_y1, target_x2, target_y2 = gt_boxes.unbind(1)
        target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
        x1_intersect, y1_intersect = torch.max(pred_x1, target_x1), torch.max(pred_y1, target_y1)
        x2_intersect, y2_intersect = torch.min(pred_x2, target_x2), torch.min(pred_y2, target_y2)
        area_intersect = torch.zeros(pred_x1.size()).to(pred)
        mask = (y2_intersect > y1_intersect) * (x2_intersect > x1_intersect)
        area_intersect[mask] = (x2_intersect[mask] - x1_intersect[mask]) * (y2_intersect[mask] - y1_intersect[mask])
        x1_enclosing, y1_enclosing = torch.min(pred_x1, target_x1), torch.min(pred_y1, target_y1)
        x2_enclosing, y2_enclosing = torch.max(pred_x2, target_x2), torch.max(pred_y2, target_y2)
        area_enclosing = (x2_enclosing - x1_enclosing) * (y2_enclosing - y1_enclosing) + 1e-7
        area_union = pred_area + target_area - area_intersect + 1e-7
        ious = area_intersect / area_union
        gious = ious - (area_enclosing - area_union) / area_enclosing
        losses = 1 - gious
        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum()
        assert losses.numel() != 0
        return losses.sum()

    @torch.no_grad()
    def prepare_targets(self, targets, anchors_per_level):
        """targets: list of {"labels" 0..C-1, "boxes" XYXY}; returns labels 1..C (0 = background) and deltas."""
        anchors_per_im = torch.cat(anchors_per_level, 0)
        num_anchors_per_level = [len(a) for a in anchors_per_level]
        cls_labels, reg_targets = [], []
        for t in targets:
            bboxes_per_im, labels_per_im = t["boxes"], t["labels"] + 1
            num_gt = bboxes_per_im.shape[0]
            if num_gt == 0:                    # (the official code assumes >= 1 GT per image)
                cls_labels.append(torch.zeros(len(anchors_per_im), dtype=torch.long, device=anchors_per_im.device))
                reg_targets.append(torch.zeros_like(anchors_per_im))
                continue
            ious = pairwise_iou(anchors_per_im, bboxes_per_im)                                  # boxlist_iou
            gt_cx = (bboxes_per_im[:, 2] + bboxes_per_im[:, 0]) / 2.0
            gt_cy = (bboxes_per_im[:, 3] + bboxes_per_im[:, 1]) / 2.0
            gt_points = torch.stack((gt_cx, gt_cy), dim=1)
            anchors_cx_per_im = (anchors_per_im[:, 2] + anchors_per_im[:, 0]) / 2.0
            anchors_cy_per_im = (anchors_per_im[:, 3] + anchors_per_im[:, 1]) / 2.0
            anchor_points = torch.stack((anchors_cx_per_im, anchors_cy_per_im), dim=1)
            distances = (anchor_points[:, None, :] - gt_points[None, :, :]).pow(2).sum(-1).sqrt()
            candidate_idxs, star_idx = [], 0
            for level in range(len(anchors_per_level)):
                end_idx = star_idx + num_anchors_per_level[level]
                topk = min(self.topk * self.num_anchors_per_loc, num_anchors_per_level[level])
                _, topk_idxs_per_level = distances[star_idx:end_idx, :].topk(topk, dim=0, largest=False)
                candidate_idxs.append(topk_idxs_per_level + star_idx)
                star_idx = end_idx
            candidate_idxs = torch.cat(candidate_idxs, dim=0)
            candidate_ious = ious[candidate_idxs, torch.arange(num_gt)]
            iou_thresh_per_gt = candidate_ious.mean(0) + candidate_ious.std(0)
            is_pos = candidate_ious >= iou_thresh_per_gt[None, :]
            anchor_num = anchors_cx_per_im.shape[0]
            for ng in range(num_gt):
                candidate_idxs[:, ng] += ng * anchor_num
            e_anchors_cx = anchors_cx_per_im.view(1, -1).expand(num_gt, anchor_num).contiguous().view(-1)
            e_anchors_cy = anchors_cy_per_im.view(1, -1).expand(num_gt, anchor_num).contiguous().view(-1)
            candidate_idxs = candidate_idxs.view(-1)
            l = e_anchors_cx[candidate_idxs].view(-1, num_gt) - bboxes_per_im[:, 0]
            t_ = e_anchors_cy[candidate_idxs].view(-1, num_gt) - bboxes_per_im[:, 1]
            r = bboxes_per_im[:, 2] - e_anchors_cx[candidate_idxs].view(-1, num_gt)
            b = bboxes_per_im[:, 3] - e_anchors_cy[candidate_idxs].view(-1, num_gt)
            is_in_gts = torch.stack([l, t_, r, b], dim=1).min(dim=1)[0] > 0.01
            is_pos = is_pos & is_in_gts
            ious_inf = torch.full_like(ious, -INF).t().contiguous().view(-1)
            index = candidate_idxs.view(-1)[is_pos.view(-1)]
            ious_inf[index] = ious.t().contiguous().view(-1)[index]
            ious_inf = ious_inf.view(num_gt, -1).t()
            anchors_to_gt_values, anchors_to_gt_indexs = ious_inf.max(dim=1)
            cls_labels_per_im = labels_per_im[anchors_to_gt_indexs].clone()
            cls_labels_per_im[anchors_to_gt_values == -INF] = 0
            matched_gts = bboxes_per_im[anchors_to_gt_indexs]
            cls_labels.append(cls_labels_per_im)
            reg_targets.append(self.box_coder.encode(matched_gts, anchors_per_im))
        return cls_labels, reg_targets

    def compute_centerness_targets(self, reg_targets, anchors):
        gts = self.box_coder.decode(reg_targets, anchors)
        anchors_cx = (anchors[:, 2] + anchors[:, 0]) / 2
        anchors_cy = (anchors[:, 3] + anchors[:, 1]) / 2
        left_right = torch.stack([anchors_cx - gts[:, 0], gts[:, 2] - anchors_cx], dim=1)
        top_bottom = torch.stack([anchors_cy - gts[:, 1], gts[:, 3] - anchors_cy], dim=1)
        centerness = torch.sqrt((left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0])
                                * (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0]))
        assert not torch.isnan(centerness).any()
        return centerness

    def __call__(self, box_cls, box_regression, centerness, targets, anchors_per_level):
        """box_cls [B, R, C], box_regression [B, R, 4], centerness [B, R] (flattened like concat_box_prediction_layers)"""
        labels, reg_targets = self.prepare_targets(targets, anchors_per_level)
        N = len(labels)
        anchors = torch.cat(anchors_per_level, 0)
        box_cls_flatten = box_cls.reshape(-1, self.num_classes)
        box_regression_flatten = box_regression.reshape(-1, 4)
        centerness_flatten = centerness.reshape(-1)
        labels_flatten = torch.cat(labels, dim=0)
        reg_targets_flatten = torch.cat(reg_targets, dim=0)
        anchors_flatten = anchors.repeat(N, 1)
        pos_inds = torch.nonzero(labels_flatten > 0).squeeze(1)
        num_pos_avg_per_gpu = max(float(pos_inds.numel()), 1.0)
        # SigmoidFocalLoss (labels 1..C, 0 = background) == one-hot sigmoid focal loss over classes 1..C
        cls_target = torch.nn.functional.one_hot(labels_flatten, self.num_classes + 1)[:, 1:].to(box_cls_flatten.dtype)
        cls_loss = sigmoid_focal_loss(box_cls_flatten, cls_target, 0.25, 2.0, "sum") / num_pos_avg_per_gpu
        box_regression_flatten = box_regression_flatten[pos_inds]
        reg_targets_flatten = reg_targets_flatten[pos_inds]
        anchors_flatten = anchors_flatten[pos_inds]
        centerness_flatten = centerness_flatten[pos_inds]
        if pos_inds.numel() > 0:
            centerness_targets = self.compute_centerness_targets(reg_targets_flatten, anchors_flatten)
            sum_centerness_targets_avg_per_gpu = centerness_targets.sum().item()
            reg_loss = self.GIoULoss(box_regression_flatten, reg_targets_flatten, anchors_flatten,
                                     weight=centerness_targets) / sum_centerness_targets_avg_per_gpu
            centerness_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                centerness_flatten, centerness_targets, reduction="sum") / num_pos_avg_per_gpu
        else:
            reg_loss = box_regression_flatten.sum()
            centerness_loss = centerness_flatten.sum()
        losses = {"loss_cls": cls_loss, "loss_reg": reg_loss * REG_LOSS_WEIGHT, "loss_centerness": centerness_loss}
        losses["losses"] = sum(losses.values())
        losses["num_pos"] = torch.tensor(pos_inds.numel() / N)
        return losses, labels_flatten > 0, labels_flatten[pos_inds] - 1
