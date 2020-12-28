# -*- coding: utf-8 -*-
# !@time: 2021/1/15 上午11:19
# !@author: superMC @email: 18758266469@163.com
# !@fileName: dma_fpn.py
from abc import ABC
from torch import nn

from .dma import MultiHeadSpatialSelfAttention
from torch.nn import functional as F


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


class FPNAttentionBottom(nn.Module, ABC):
    def __init__(self, in_planes, downsample_cfg='conv', bottom_conv=True, self_conv=True):
        super().__init__()
        if bottom_conv:
            self.query_bottom_conv = nn.Conv2d(in_planes, in_planes, kernel_size=1)
            self.value_bottom_conv = nn.Conv2d(in_planes, in_planes, kernel_size=1)
        if self_conv:
            self.query_self_conv = nn.Conv2d(in_planes, in_planes, kernel_size=1)
            self.value_self_conv = nn.Conv2d(in_planes, in_planes, kernel_size=1)
            self.key_self_conv = nn.Conv2d(in_planes, in_planes, kernel_size=1)
        if downsample_cfg == 'maxpool':
            self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)
        elif downsample_cfg == 'conv':
            self.downsample = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=2, padding=1)

    def query_value_bottom_conv(self, x, prev_shape=None):
        if prev_shape:
            x = F.adaptive_max_pool2d_with_indices(x, prev_shape)
        else:
            x = self.downsample(x)
        q = self.query_bottom_conv(x)
        v = self.value_bottom_conv(x)

        batch_size, channel_num, width, height = q.size()
        q = q.permute(0, 2, 3, 1).contiguous().view(batch_size, width, height, 1, channel_num)
        batch_size, channel_num, width, height = v.size()
        v = v.permute(0, 2, 3, 1).contiguous().view(batch_size, width, height, channel_num, 1)
        return q, v

    def query_value_key_self_conv(self, x):
        q = self.query_self_conv(x)
        v = self.value_self_conv(x)
        k = self.key_self_conv(x)
        batch_size, channel_num, width, height = q.size()
        q = q.permute(0, 2, 3, 1).contiguous().view(batch_size, width, height, 1, channel_num)
        batch_size, channel_num, width, height = v.size()
        v = v.permute(0, 2, 3, 1).contiguous().view(batch_size, width, height, channel_num, 1)
        batch_size, channel_num, width, height = k.size()
        k = k.permute(0, 2, 3, 1).contiguous().view(batch_size, width, height, channel_num, 1)
        return q, v, k


class FPNAttentionUp(nn.Module, ABC):
    def __init__(self, in_planes, upsample_cfg, up_conv=True, self_conv=True):
        super().__init__()
        if up_conv:
            self.query_up_conv = nn.Conv2d(in_planes, in_planes, kernel_size=1)
            self.value_up_conv = nn.Conv2d(in_planes, in_planes, kernel_size=1)
        if self_conv:
            self.query_self_conv = nn.Conv2d(in_planes, in_planes, kernel_size=1)
            self.value_self_conv = nn.Conv2d(in_planes, in_planes, kernel_size=1)
            self.key_self_conv = nn.Conv2d(in_planes, in_planes, kernel_size=1)
        self.upsample_cfg = upsample_cfg

    def query_value_up_conv(self, x, prev_shape):
        q = self.query_up_conv(x)
        v = self.value_up_conv(x)
        if 'scale_factor' in self.upsample_cfg:
            q = F.interpolate(q, **self.upsample_cfg)
            v = F.interpolate(v, **self.upsample_cfg)
        else:
            q = F.interpolate(q, size=prev_shape, **self.upsample_cfg)
            v = F.interpolate(v, size=prev_shape, **self.upsample_cfg)
        batch_size, channel_num, width, height = q.size()
        q = q.permute(0, 2, 3, 1).contiguous().view(batch_size, width, height, 1, channel_num)
        batch_size, channel_num, width, height = v.size()
        v = v.permute(0, 2, 3, 1).contiguous().view(batch_size, width, height, channel_num, 1)
        return q, v

    def query_value_key_self_conv(self, x):
        q = self.query_self_conv(x)
        v = self.value_self_conv(x)
        k = self.key_self_conv(x)
        batch_size, channel_num, width, height = q.size()
        q = q.permute(0, 2, 3, 1).contiguous().view(batch_size, width, height, 1, channel_num)
        batch_size, channel_num, width, height = v.size()
        v = v.permute(0, 2, 3, 1).contiguous().view(batch_size, width, height, channel_num, 1)
        batch_size, channel_num, width, height = k.size()
        k = k.permute(0, 2, 3, 1).contiguous().view(batch_size, width, height, channel_num, 1)
        return q, v, k
