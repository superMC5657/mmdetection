# -*- coding: utf-8 -*-
# !@time: 2020/12/23 下午9:32
# !@author: superMC @email: 18758266469@163.com
# !@fileName: attn_resnet.py
from abc import ABC

from ..resnet import ResNet
from ... import BACKBONES
from ...attn_module.dma import MultiHeadSpatialSelfAttention, MultiHeadChannelSelfAttention


@BACKBONES.register_module()
class AttnResNet(ResNet, ABC):
    def __init__(self, depth, in_channels=3, stem_channels=None, base_channels=64, num_stages=4, strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1), out_indices=(0, 1, 2, 3), style='pytorch', deep_stem=False, avg_down=False,
                 frozen_stages=-1, conv_cfg=None, norm_cfg=dict(type='BN', requires_grad=True), norm_eval=True,
                 dcn=None, stage_with_dcn=(False, False, False, False), plugins=None, with_cp=False,
                 zero_init_residual=True):
        super().__init__(depth, in_channels, stem_channels, base_channels, num_stages, strides, dilations, out_indices,
                         style, deep_stem, avg_down, frozen_stages, conv_cfg, norm_cfg, norm_eval, dcn, stage_with_dcn,
                         plugins, with_cp, zero_init_residual)
        self.msa = MultiHeadSpatialSelfAttention(self.feat_dim)
        self.mca = MultiHeadChannelSelfAttention(self.feat_dim)

    def forward(self, x):
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        attns = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        x, attn = self.mca(x)
        attns.append(attn)
        x, attn = self.msa(x)
        attns.append(attn)
        return tuple(outs), tuple(attns)
