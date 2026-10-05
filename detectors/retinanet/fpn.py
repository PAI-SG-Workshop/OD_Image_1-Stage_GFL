"""
Feature Pyramid Network with the RetinaNet P6 / P7 top block.

Code : detectron2 modeling/backbone/fpn.py (FPN, LastLevelP6P7, build_retinanet_resnet_fpn_backbone),
       configs/Base-RetinaNet.yaml (FPN.IN_FEATURES = res3, res4, res5; OUT_CHANNELS 256; NORM "";
       FUSE_TYPE "sum"). The detectron2 Backbone / ShapeSpec / registry interfaces are removed.
Paper: Lin et al., "Focal Loss for Dense Object Detection" (ICCV 2017), Sec. 4: P3-P5 from the FPN,
       P6 = 3x3 stride-2 conv on C5, P7 = ReLU + 3x3 stride-2 conv on P6.

Weight init: detectron2 uses c2_xavier_fill; the framework's common init (weights_init_normal) replaces
the conv weights afterwards ([Unified]); conv biases keep detectron2's 0.
"""

import torch.nn as nn
import torch.nn.functional as F


def c2_xavier_fill(module):
    """detectron2 fvcore.nn.weight_init.c2_xavier_fill"""
    nn.init.kaiming_uniform_(module.weight, a=1)
    if module.bias is not None:
        nn.init.constant_(module.bias, 0)


class LastLevelP6P7(nn.Module):
    """P6 and P7 from the C5 feature (detectron2 LastLevelP6P7, in_feature="res5")."""

    def __init__(self, in_channels, out_channels, in_feature="res5", relu_before_p7=True):
        super().__init__()
        self.num_levels = 2
        self.relu_before_p7 = relu_before_p7      # mmdet FPN(add_extra_convs="on_output"): no ReLU (GFL)
        self.in_feature = in_feature
        self.p6 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        self.p7 = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
        for module in [self.p6, self.p7]:
            c2_xavier_fill(module)

    def forward(self, c5):
        p6 = self.p6(c5)
        p7 = self.p7(F.relu(p6) if self.relu_before_p7 else p6)
        return [p6, p7]


class FPN(nn.Module):
    """detectron2 FPN (norm "", fuse "sum"): forward(bottom-up dict) -> [p3, p4, p5, p6, p7]."""

    def __init__(self, in_channels, out_channels=256, in_features=("res3", "res4", "res5"), top_block=None):
        super().__init__()
        self.in_features = list(in_features)
        lateral_convs, output_convs = [], []
        for c in [in_channels[f] for f in self.in_features]:
            lateral_conv = nn.Conv2d(c, out_channels, kernel_size=1, bias=True)
            output_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
            c2_xavier_fill(lateral_conv)
            c2_xavier_fill(output_conv)
            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)
        # detectron2 stores them top-down (res5 first)
        self.lateral_convs = nn.ModuleList(lateral_convs[::-1])
        self.output_convs = nn.ModuleList(output_convs[::-1])
        self.top_block = top_block

    def forward(self, bottom_up_features):
        results = []
        prev_features = self.lateral_convs[0](bottom_up_features[self.in_features[-1]])
        results.append(self.output_convs[0](prev_features))
        for idx, (lateral_conv, output_conv) in enumerate(zip(self.lateral_convs, self.output_convs)):
            if idx > 0:
                features = bottom_up_features[self.in_features[-idx - 1]]
                top_down_features = F.interpolate(prev_features, scale_factor=2.0, mode="nearest")
                prev_features = lateral_conv(features) + top_down_features
                results.insert(0, output_conv(prev_features))
        if self.top_block is not None:
            # detectron2: the top block reads a bottom-up feature ("res5", RetinaNet) or an FPN output ("p5", FCOS)
            if self.top_block.in_feature in bottom_up_features:
                top_block_in_feature = bottom_up_features[self.top_block.in_feature]
            else:
                top_block_in_feature = results[-1]          # "p5": the coarsest FPN output
            results.extend(self.top_block(top_block_in_feature))
        return results


def build_retinanet_fpn(bottom_up_channels, out_channels=256, top_block_in="res5", relu_before_p7=True):
    """detectron2 build_retinanet_resnet_fpn_backbone (neck part): P6 / P7 from res5 (RetinaNet)
    or from p5 (FCOS: configs/common/models/fcos.py; ATSS: RETINANET.USE_C5 False)."""
    in_ch = bottom_up_channels["res5"] if top_block_in == "res5" else out_channels
    return FPN(bottom_up_channels, out_channels, top_block=LastLevelP6P7(in_ch, out_channels, top_block_in, relu_before_p7))
