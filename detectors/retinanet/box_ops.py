"""
Anchors, anchor matching and box coding for RetinaNet.

Code : detectron2
    modeling/anchor_generator.py  DefaultAnchorGenerator (generate_cell_anchors, _create_grid_offsets,
                                  _grid_anchors); OFFSET 0.0; ASPECT_RATIOS [0.5, 1.0, 2.0] (H/W)
    configs/Base-RetinaNet.yaml   SIZES [[x, x*2^(1/3), x*2^(2/3)] for x in 32, 64, 128, 256, 512]
    modeling/matcher.py           Matcher (IOU_THRESHOLDS [0.4, 0.5], IOU_LABELS [0, -1, 1],
                                  allow_low_quality_matches=True)
    modeling/box_regression.py    Box2BoxTransform (weights 1, 1, 1, 1; scale_clamp log(1000 / 16))
    structures/boxes.py           pairwise_iou
Paper: Lin et al. 2017, Sec. 4 "Anchors": areas 32^2..512^2 on P3..P7, 3 ratios x 3 scales = A = 9;
       IoU >= 0.5 foreground, [0, 0.4) background, [0.4, 0.5) ignored.
"""


import torch


def pairwise_iou(boxes1, boxes2):
    """detectron2 structures/boxes.py pairwise_iou: [M, 4] x [N, 4] XYXY -> [M, N]."""
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    width_height = torch.min(boxes1[:, None, 2:], boxes2[:, 2:]) - torch.max(boxes1[:, None, :2], boxes2[:, :2])
    width_height.clamp_(min=0)
    inter = width_height.prod(dim=2)
    return torch.where(inter > 0, inter / (area1[:, None] + area2 - inter),
                       torch.zeros(1, dtype=inter.dtype, device=inter.device))
