# -*- coding: utf-8 -*-
# !@time: 2020/12/31 下午10:03
# !@author: superMC @email: 18758266469@163.com
# !@fileName: attn_fpn.py
import warnings
from abc import ABC

import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, xavier_init
from mmcv.runner import auto_fp16

from ...attn_module.dma import MultiHeadSpatialSelfAttention
from ...builder import NECKS
from ... import FPN
from torch import nn, matmul


class AttnFPNv1(FPN, ABC):

    def __init__(self, in_channels, out_channels, num_outs, *args, **kwargs):
        r"""Feature Pyramid Network with attention"""
        super().__init__(in_channels, out_channels, num_outs, *args, **kwargs)

    @auto_fp16()
    def forward(self, inputs):
        batch_size = inputs[0].size(0)
        sizes = []
        for i in range(len(inputs)):
            sizes.append(inputs[i].size()[2:])
        """Forward function"""
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level]).reshape(batch_size, self.out_channels, -1)
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        # build top-down path
        used_backbone_levels = len(laterals)
        outs = []
        for i in range(used_backbone_levels):
            for j in range(used_backbone_levels):
                attn_features = matmul(matmul(laterals[i].transpose(1, 2).contiguous(), laterals[j]),
                                       laterals[j].transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
                if j == 0:
                    outs.append(attn_features)
                else:
                    outs[i] += attn_features
        outs = [outs[i] + laterals[i] for i in range(used_backbone_levels)]
        outs = [self.fpn_convs[i](outs[i].view(batch_size, self.out_channels, sizes[i][0], sizes[i][1])) for i in
                range(used_backbone_levels)]

        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)


class MultiHeadSpatialFPNAttention(MultiHeadSpatialSelfAttention, ABC):
    def __init__(self, in_planes, hidden_state=16, multi_head=4, alpha=0.5, fuse=False):
        super().__init__(in_planes, hidden_state, multi_head, alpha, fuse)

    def forward(self, inputs):
        x1, x2 = inputs
        batch_size_1, channel_num_1, width_1, height_1 = x1.size()
        batch_size_2, channel_num_2, width_2, height_2 = x2.size()
        proj_query = self.query_conv(x1).view(batch_size_1, self.multi_head, -1, width_1 * height_1).transpose(2,
                                                                                                               3).contiguous()
        proj_key = self.key_conv(x2).view(batch_size_2, self.multi_head, -1, width_2 * height_2)
        out = self.value_conv(x2).view(batch_size_2, self.multi_head, -1, width_2 * height_2).transpose(2,
                                                                                                        3).contiguous()

        out, attn = self.attention(proj_query, proj_key, out)
        out = out.transpose(2, 3).contiguous().view(batch_size_1, -1, width_1, height_1)
        out = self.conv(out)
        out = self.bn(out)

        return out, attn
