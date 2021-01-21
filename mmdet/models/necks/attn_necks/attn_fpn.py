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
from ...attn_module.dma_fpn import FPNAttentionUp


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


class AttnFPN(FPN, ABC):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest')):
        super(AttnFPN,
              self).__init__(in_channels, out_channels, num_outs, start_level,
                             end_level, add_extra_convs, extra_convs_on_inputs,
                             relu_before_extra_convs, no_norm_on_lateral,
                             conv_cfg, norm_cfg, act_cfg, upsample_cfg)
        self.fpn_attn_up = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level):
            up_conv = True
            self_conv = True
            if i == self.start_level:
                up_conv = False
            if i == self.backbone_end_level - 1:
                self_conv = False
            self.fpn_attn_up.append(
                FPNAttentionUp(out_channels, upsample_cfg=self.upsample_cfg, up_conv=up_conv, self_conv=self_conv))
        self.attention = ScaledDotProductAttention(out_channels ** 0.5)

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
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            query_up, value_up = self.fpn_attn_up[i].query_value_up_conv(laterals[i], prev_shape)
            query_self, value_self, key_self = self.fpn_attn_up[i - 1].query_value_key_self_conv(laterals[i - 1])
            query = torch.cat((query_up, query_self), dim=-2)
            value = torch.cat((value_up, value_self), dim=-1)
            laterals[i - 1] = self.attention(query, value, key_self)

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

    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=-2)

    def forward(self, q, v, k):
        attn = matmul(q / self.temperature, k)
        attn = self.softmax(attn)
        v = matmul(v, attn).squeeze(dim=-1)
        batch_size, width, height, channel_num = v.size()
        v = v.permute(0, 3, 1, 2).contiguous().view(batch_size, channel_num, width, height)
        return v
