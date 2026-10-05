"""
GFL (Generalized Focal Loss) builder, Image task only.

Paper: Li et al., "Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object
       Detection" (NeurIPS 2020, arXiv 2006.04388). Origin of the Distribution Focal Loss (DFL) used by YOLOv6 / v8.
Code : official implus/GFocal is an mmdetection 2.x fork; the same classes in mmdetection 3.x are used:
       configs/gfl/gfl_r50_fpn_1x_coco.py, models/dense_heads/gfl_head.py (GFLHead, Integral),
       losses/gfocal_loss.py (QualityFocalLoss beta 2, DistributionFocalLoss 0.25), GIoULoss 2.0,
       task_modules/assigners/atss_assigner.py (ATSSAssigner topk 9), coders/distance_point_bbox_coder.py,
       AnchorGenerator(ratios [1], octave_base_scale 8, center_offset 0).
       Backbone / FPN shared with detectors/retinanet (ResNet-50, FPN P3-P7, extra convs on P5 output without
       ReLU = mmdet add_extra_convs='on_output'). ATSS assignment reuses detectors/atss (same algorithm as the
       mmdet ATSSAssigner).

    head : 4 x (3x3 conv + GN + ReLU) per branch, gfl_cls (C), gfl_reg (4 x 17 bins), learnable Scale per level
    box  : distances = Integral(softmax(17 bins)) x stride from the anchor centre
    loss : QFL (target = IoU of the decoded positive box) / #pos + GIoU and DFL weighted by max cls score
           / sum(weights)
Raw contract: sigmoid class probabilities (= QFL joint quality-class score), objectness = 1.

[Unified] as RetinaNet / FCOS / ATSS: no ImageNet weights, trainable BN, common init (head GN and prior bias
kept), no mean / std, 608 input, Adam 1e-4, common post-processing, PreciseBN.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from detectors.anchor_free.common import HeadStats, split_targets, to_raw
from detectors.atss.build import ATSSLoss, Scale
from detectors.retinanet.fpn import build_retinanet_fpn
from detectors.retinanet.resnet import build_resnet
from detectors.rtmdet.build import bbox_overlaps, quality_focal_loss

STRIDES = (8, 16, 32, 64, 128)
REG_MAX = 16


class ConvGN(nn.Module):
    """mmcv ConvModule(norm_cfg=GN(32), act ReLU): conv without bias (bias='auto' with a norm layer)."""

    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, 1, 1, bias=False)
        self.gn = nn.GroupNorm(32, cout)
        self.activate = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.activate(self.gn(self.conv(x)))


class Integral(nn.Module):
    """mmdet gfl_head.Integral: expectation over reg_max + 1 bins."""

    def __init__(self, reg_max=REG_MAX):
        super().__init__()
        self.reg_max = reg_max
        self.register_buffer("project", torch.linspace(0, self.reg_max, self.reg_max + 1), persistent=False)

    def forward(self, x):
        x = F.softmax(x.reshape(-1, self.reg_max + 1), dim=1)
        return F.linear(x, self.project.type_as(x)).reshape(-1, 4)


def distance2bbox(points, distance):
    return torch.stack([points[..., 0] - distance[..., 0], points[..., 1] - distance[..., 1],
                        points[..., 0] + distance[..., 2], points[..., 1] + distance[..., 3]], -1)


def bbox2distance(points, bbox, max_dis=None, eps=0.1):
    """mmdet structures/bbox/transforms.bbox2distance"""
    left, top = points[:, 0] - bbox[:, 0], points[:, 1] - bbox[:, 1]
    right, bottom = bbox[:, 2] - points[:, 0], bbox[:, 3] - points[:, 1]
    if max_dis is not None:
        left, top = left.clamp(min=0, max=max_dis - eps), top.clamp(min=0, max=max_dis - eps)
        right, bottom = right.clamp(min=0, max=max_dis - eps), bottom.clamp(min=0, max=max_dis - eps)
    return torch.stack([left, top, right, bottom], -1)


def distribution_focal_loss(pred, label):
    """mmdet gfocal_loss.distribution_focal_loss (per sample)"""
    dis_left = label.long()
    dis_right = dis_left + 1
    weight_left = dis_right.float() - label
    weight_right = label - dis_left.float()
    return F.cross_entropy(pred, dis_left, reduction="none") * weight_left + \
        F.cross_entropy(pred, dis_right, reduction="none") * weight_right


class _IdentityCoder:
    """lets ATSSLoss.prepare_targets return the matched GT boxes instead of encoded deltas"""

    def encode(self, gt_boxes, anchors):
        return gt_boxes


def mmdet_anchors(sizes_hw, device, octave_base_scale=8):
    """mmdet AnchorGenerator(ratios=[1], octave_base_scale=8, scales_per_octave=1, center_offset=0)"""
    out = []
    for (h, w), s in zip(sizes_hw, STRIDES):
        half = octave_base_scale * s / 2
        sy, sx = torch.meshgrid(torch.arange(h, device=device, dtype=torch.float32) * s,
                                torch.arange(w, device=device, dtype=torch.float32) * s, indexing="ij")
        sx, sy = sx.reshape(-1), sy.reshape(-1)
        out.append(torch.stack([sx - half, sy - half, sx + half, sy + half], 1))
    return out


def atss_assign(anchors_lv, targets, num_classes, topk=9):
    """mmdet ATSSAssigner -> labels [B, R] (num_classes = background), matched GT boxes [B, R, 4]"""
    a = ATSSLoss(num_classes, _IdentityCoder(), topk=topk)
    labels, boxes = a.prepare_targets(targets, anchors_lv)            # labels 1..C, 0 = background
    labels = torch.stack(labels) - 1
    labels[labels < 0] = num_classes
    return labels, torch.stack(boxes)


class GFLHead(nn.Module):
    def __init__(self, num_classes, in_channels=256, feat_channels=256, stacked_convs=4, reg_max=REG_MAX):
        super().__init__()
        self.cls_convs = nn.ModuleList([ConvGN(in_channels if i == 0 else feat_channels, feat_channels)
                                        for i in range(stacked_convs)])
        self.reg_convs = nn.ModuleList([ConvGN(in_channels if i == 0 else feat_channels, feat_channels)
                                        for i in range(stacked_convs)])
        self.gfl_cls = nn.Conv2d(feat_channels, num_classes, 3, padding=1)
        self.gfl_reg = nn.Conv2d(feat_channels, 4 * (reg_max + 1), 3, padding=1)
        self.scales = nn.ModuleList([Scale(1.0) for _ in STRIDES])
        for m in self.modules():                       # init_cfg: Normal(std 0.01), gfl_cls bias prior 0.01
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.constant_(self.gfl_cls.bias, -math.log((1 - 0.01) / 0.01))

    def forward(self, feats):
        cls_scores, bbox_preds = [], []
        for x, scale in zip(feats, self.scales):
            cls_feat, reg_feat = x, x
            for c in self.cls_convs:
                cls_feat = c(cls_feat)
            for c in self.reg_convs:
                reg_feat = c(reg_feat)
            cls_scores.append(self.gfl_cls(cls_feat))
            bbox_preds.append(scale(self.gfl_reg(reg_feat)).float())
        return cls_scores, bbox_preds


class GFL(nn.Module):
    def __init__(self, num_classes, depth=50):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = build_resnet(depth)
        self.fpn = build_retinanet_fpn(self.backbone.out_channels, 256, top_block_in="p5", relu_before_p7=False)
        self.head = GFLHead(num_classes)
        self.integral = Integral(REG_MAX)
        self.precise_bn = True
        self.precise_bn_images = 1600
        self.precise_bn_batch_size = 16
        self.stats = HeadStats()
        self.yolo_layers = [self.stats]

    def forward(self, x, targets=None):
        cls_scores, bbox_preds = self.head(self.fpn(self.backbone(x)))
        B, C = x.shape[0], self.num_classes
        anchors_lv = mmdet_anchors([c.shape[-2:] for c in cls_scores], x.device)
        anchors = torch.cat(anchors_lv, 0)
        centers = (anchors[:, :2] + anchors[:, 2:]) / 2
        strides = torch.cat([torch.full((len(a),), s, device=x.device) for a, s in zip(anchors_lv, STRIDES)])
        cls = torch.cat([c.permute(0, 2, 3, 1).reshape(B, -1, C) for c in cls_scores], 1)
        reg = torch.cat([b.permute(0, 2, 3, 1).reshape(B, -1, 4 * (REG_MAX + 1)) for b in bbox_preds], 1)
        dist = self.integral(reg).view(B, -1, 4) * strides[None, :, None]
        boxes = distance2bbox(centers[None], dist)
        raw = to_raw(boxes, cls.sigmoid())
        if targets is None:
            return raw
        labels, gt_boxes = atss_assign(anchors_lv, split_targets(targets, B, x.shape[2]), C)
        labels_f = labels.view(-1)
        pos = (labels_f < C).nonzero().squeeze(1)
        cls_f = cls.reshape(-1, C)
        score = cls_f.new_zeros(labels_f.shape)
        num_pos = max(float(len(pos)), 1.0)
        if len(pos) > 0:
            st = strides.repeat(B)[pos]
            pos_centers = centers.repeat(B, 1)[pos] / st[:, None]
            pos_reg = reg.reshape(-1, 4 * (REG_MAX + 1))[pos]
            weight_targets = cls_f.detach().sigmoid().max(dim=1)[0][pos]
            pos_decode_pred = distance2bbox(pos_centers, self.integral(pos_reg))
            pos_decode_target = gt_boxes.view(-1, 4)[pos] / st[:, None]
            score[pos] = bbox_overlaps(pos_decode_pred.detach(), pos_decode_target, is_aligned=True)
            target_corners = bbox2distance(pos_centers, pos_decode_target, REG_MAX, eps=0.01).reshape(-1)
            giou = bbox_overlaps(pos_decode_pred, pos_decode_target, mode="giou", is_aligned=True)
            loss_bbox = 2.0 * ((1 - giou) * weight_targets).sum()
            loss_dfl = 0.25 * (distribution_focal_loss(pos_reg.reshape(-1, REG_MAX + 1), target_corners)
                               * weight_targets[:, None].expand(-1, 4).reshape(-1)).sum() / 4.0
            w_avg = weight_targets.sum().clamp(min=1).item()
        else:
            loss_bbox = loss_dfl = reg.sum() * 0
            w_avg = 1.0
        loss_cls = quality_focal_loss(cls_f, labels_f, score).sum() / num_pos
        loss_dict = {"loss_cls": loss_cls, "loss_bbox": loss_bbox / w_avg, "loss_dfl": loss_dfl / w_avg}
        loss_dict["losses"] = sum(loss_dict.values())
        loss_dict["num_pos"] = torch.tensor(len(pos) / B)
        pos_mask = labels_f < C
        self.stats.update(loss_dict, cls_f.detach().sigmoid().max(-1).values, cls_f.detach(), pos_mask,
                          labels_f[pos_mask])
        return loss_dict["losses"], raw

    def load_darknet_weights(self, path):
        raise NotImplementedError("GFL has no Darknet weights; train from scratch.")


def build_model(task, num_classes):
    return GFL(num_classes)
