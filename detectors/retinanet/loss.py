"""
RetinaNet loss: sigmoid focal loss + L1 box regression, normalised by an EMA of the foreground count.

Code : detectron2 modeling/meta_arch/retinanet.py (RetinaNet.losses, RetinaNet.label_anchors),
       dense_detector.py (_ema_update: momentum 0.9, initial 100), box_regression.py
       (_dense_box_regression_loss, "smooth_l1" with Base-RetinaNet.yaml SMOOTH_L1_LOSS_BETA 0.0 = L1),
       fvcore.nn.sigmoid_focal_loss (alpha 0.25, gamma 2, reduction "sum").
Paper: Lin et al. 2017, Eq. 5 (alpha-balanced focal loss) and Sec. 4: "the total focal loss of an image
       is computed as the sum of the focal loss over all ~100k anchors, normalized by the number of
       anchors assigned to a ground-truth box"; ignored anchors (IoU 0.4-0.5) get no loss.
"""

import torch
import torch.nn.functional as F


def sigmoid_focal_loss(inputs, targets, alpha=0.25, gamma=2.0, reduction="sum"):
    """fvcore.nn.focal_loss.sigmoid_focal_loss"""
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    return loss
