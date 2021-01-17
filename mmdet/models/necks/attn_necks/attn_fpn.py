# -*- coding: utf-8 -*-
# !@time: 2020/12/31 下午10:03
# !@author: superMC @email: 18758266469@163.com
# !@fileName: attn_fpn.py
from abc import ABC

import torch
import torch.nn.functional as F
from mmcv.runner import auto_fp16
from torch import matmul, nn

from ... import FPN
from ...attn_module.dma_fpn import FPNAttentionV2


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
                # 参数爆炸 因为sum没有做归一化
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


class AttnFPNv2(FPN, ABC):
    def __init__(self, in_channels, out_channels, num_outs, *args, **kwargs):
        super().__init__(in_channels, out_channels, num_outs, *args, **kwargs)
        self.fpn_attn = nn.ModuleList()
        for i in range(num_outs):
            first = False
            last = False
            if i == 0:
                first = False
            if i == num_outs - 1:
                last = False
            self.fpn_attn.append(
                FPNAttentionV2(out_channels, upsample_cfg=self.upsample_cfg, first=first, last=last))
        self.attention = ScaledDotProductAttention(out_channels)

    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        attns_1 = []
        for i in range(used_backbone_levels - 1, 0, -1):
            k_i = self.fpn_attn[i].k_up_forward(laterals[i], prev_shape=laterals[i - 1].shape[2:])
            q_i_next = self.fpn_attn[i - 1].q_1_forward(laterals[i - 1])
            k_i_next = self.fpn_attn[i - 1].k_self_1_forward(laterals[i - 1])
            v_i_next = self.fpn_attn[i - 1].v_1_forward(laterals[i - 1])
            k = torch.cat((k_i, k_i_next), dim=-2)
            attn, laterals[i - 1] = self.attention(q_i_next, k, v_i_next)
            attns_1.append(attn)
        attns_2 = []
        for i in range(0, used_backbone_levels - 1):
            k_i = self.fpn_attn[i].k_down_forward(laterals[i])
            q_i_next = self.fpn_attn[i + 1].q_2_forward(laterals[i + 1])
            k_i_next = self.fpn_attn[i + 1].k_self_1_forward(laterals[i + 1])
            v_i_next = self.fpn_attn[i + 1].v_2_forward(laterals[i + 1])
            k = torch.cat((k_i, k_i_next), dim=-2)
            attn, laterals[i + 1] = self.attention(q_i_next, k, v_i_next)
            attns_2.append(attn)
        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
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


class ScaledDotProductAttention(nn.Module, ABC):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        # self.dropout = nn.Dropout(attn_dropout, inplace=False)
        self.softmax = nn.Softmax(dim=-2)

    def forward(self, q, k, v):
        attn = matmul(q / self.temperature, k)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        v = matmul(attn, v)
        v = torch.sum(v, dim=-2)
        if len(v.size()) == 5:
            v = v.squeeze(dim=-1)
        return v, attn
