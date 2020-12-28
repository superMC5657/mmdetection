# -*- coding: utf-8 -*-
# !@time: 2021/1/20 下午12:18
# !@author: superMC @email: 18758266469@163.com
# !@fileName: attn_pafpn.py
from abc import ABC

import torch
from mmcv.cnn import ConvModule
from mmcv.runner import auto_fp16
import torch.nn.functional as F
from torch import nn

from .attn_fpn import AttnFPN
from ...builder import NECKS
from ...attn_module import FPNAttentionBottom


@NECKS.register_module()
class AttnPAFPN(AttnFPN, ABC):
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
                 upsample_cfg=dict(mode='nearest'),
                 downsample_cfg='maxpool'):
        super(AttnPAFPN,
              self).__init__(in_channels, out_channels, num_outs, start_level,
                             end_level, add_extra_convs, extra_convs_on_inputs,
                             relu_before_extra_convs, no_norm_on_lateral,
                             conv_cfg, norm_cfg, act_cfg, upsample_cfg)
        self.fpn_attn_bottom = nn.ModuleList()
        self.pafpn_convs = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level):
            bottom_conv = True
            self_conv = True
            if i == self.start_level:
                self_conv = False
            if i == self.backbone_end_level - 1:
                bottom_conv = False
            self.fpn_attn_bottom.append(
                FPNAttentionBottom(out_channels, downsample_cfg=downsample_cfg, bottom_conv=bottom_conv,
                                   self_conv=self_conv))
            self.pafpn_convs.append(ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False))

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
        inter_outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]

        #   part 2: add bottom-up path
        for i in range(0, used_backbone_levels - 1):
            query_bottom, value_bottom = self.fpn_attn_bottom[i].query_value_bottom_conv(inter_outs[i])
            query_self, value_self, key_self = self.fpn_attn_bottom[i + 1].query_value_key_self_conv(inter_outs[i + 1])
            query = torch.cat((query_bottom, query_self), dim=-2)
            value = torch.cat((value_bottom, value_self), dim=-1)
            inter_outs[i + 1] = self.attention(query, value, key_self)
        outs = [self.pafpn_convs[i](inter_outs[i]) for i in range(self.start_level, self.backbone_end_level)]

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
