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


class FPNAttentionV2(nn.Module, ABC):
    def __init__(self, in_planes, upsample_cfg, first=False, last=False):
        super().__init__()
        self.query_1_conv = nn.Conv2d(in_planes, in_planes, kernel_size=1)
        self.query_2_conv = nn.Conv2d(in_planes, in_planes, kernel_size=1)
        if not first:
            self.key_self_1_conv = nn.Conv2d(in_planes, in_planes, kernel_size=1)
        if not last:
            self.key_self_2_conv = nn.Conv2d(in_planes, in_planes, kernel_size=1)

        self.key_up_conv = nn.Conv2d(in_planes, in_planes, kernel_size=1)
        self.key_down_conv = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=2, padding=1)

        self.value_1_conv = nn.Conv2d(in_planes, in_planes, kernel_size=1)
        self.value_2_conv = nn.Conv2d(in_planes, in_planes, kernel_size=1)
        self.upsample_cfg = upsample_cfg

    def q_1_forward(self, x):
        q = self.query_1_conv(x)
        batch_size, channel_num, width, height = q.size()
        q = q.permute(0, 2, 3, 1).contiguous().view(batch_size, width, height, 1, channel_num)
        return q

    def q_2_forward(self, x):
        q = self.query_2_conv(x)
        batch_size, channel_num, width, height = q.size()
        q = q.permute(0, 2, 3, 1).contiguous().view(batch_size, width, height, 1, channel_num)
        return q

    def k_up_forward(self, x, prev_shape):
        k = self.key_up_conv(x)
        if 'scale_factor' in self.upsample_cfg:
            k = F.interpolate(k, **self.upsample_cfg)
        else:
            k = F.interpolate(k, size=prev_shape, **self.upsample_cfg)
        batch_size, channel_num, width, height = k.size()
        k = k.permute(0, 2, 3, 1).contiguous().view(batch_size, width, height, channel_num, 1)
        return k

    def k_self_1_forward(self, x):
        k = self.key_self_1_conv(x)
        batch_size, channel_num, width, height = k.size()
        k = k.permute(0, 2, 3, 1).contiguous().view(batch_size, width, height, channel_num, 1)
        return k

    def k_self_2_forward(self, x):
        k = self.key_self_2_conv(x)
        batch_size, channel_num, width, height = k.size()
        k = k.permute(0, 2, 3, 1).contiguous().view(batch_size, width, height, channel_num, 1)
        return k

    def k_down_forward(self, x):
        k = self.key_down_conv(x)
        batch_size, channel_num, width, height = k.size()
        k = k.permute(0, 2, 3, 1).contiguous().view(batch_size, width, height, channel_num, 1)
        return k

    def v_1_forward(self, x):
        v = self.value_1_conv(x)
        batch_size, channel_num, width, height = v.size()
        v = v.permute(0, 2, 3, 1).contiguous().view(batch_size, width, height, 1, channel_num)
        return v

    def v_2_forward(self, x):
        v = self.value_2_conv(x)
        batch_size, channel_num, width, height = v.size()
        v = v.permute(0, 2, 3, 1).contiguous().view(batch_size, width, height, 1, channel_num)
        return v


class FPNAttentionV3(ABC):
    def __init__(self, in_planes, upsample_cfg, first=False, last=False):
        super().__init__()
        self.query_1_conv = nn.Conv2d(in_planes, in_planes, kernel_size=1)
        self.query_2_conv = nn.Conv2d(in_planes, in_planes, kernel_size=1)
        self.key_self_1_conv = nn.Conv2d(in_planes, in_planes, kernel_size=1)
        self.key_self_2_conv = nn.Conv2d(in_planes, in_planes, kernel_size=1)
        self.key_up_conv = nn.Conv2d(in_planes, in_planes, kernel_size=1)
        self.key_down_conv = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=2, padding=1)
        self.value_1_conv = nn.Conv2d(in_planes, in_planes, kernel_size=1)
        self.value_2_conv = nn.Conv2d(in_planes, in_planes, kernel_size=1)
        self.upsample_cfg = upsample_cfg

    def q_1_forward(self, x):
        q = self.query_1_conv(x)
        batch_size, channel_num, width, height = q.size()
        q = q.permute(0, 2, 3, 1).contiguous().view(batch_size, width, height, channel_num, 1, 1)
        return q

    def q_2_forward(self, x):
        q = self.query_2_conv(x)
        batch_size, channel_num, width, height = q.size()
        q = q.permute(0, 2, 3, 1).contiguous().view(batch_size, width, height, channel_num, 1, 1)
        return q

    def k_up_forward(self, x, prev_shape):
        k = self.key_up_conv(x)
        if 'scale_factor' in self.upsample_cfg:
            k = F.interpolate(k, **self.upsample_cfg)
        else:
            k = F.interpolate(k, size=prev_shape, **self.upsample_cfg)
        batch_size, channel_num, width, height = k.size()
        k = k.permute(0, 2, 3, 1).contiguous().view(batch_size, width, height, channel_num, 1, 1)
        return k

    def k_self_1_forward(self, x):
        k = self.key_self_1_conv(x)
        batch_size, channel_num, width, height = k.size()
        k = k.permute(0, 2, 3, 1).contiguous().view(batch_size, width, height, channel_num, 1, 1)
        return k

    def k_self_2_forward(self, x):
        k = self.key_self_2_conv(x)
        batch_size, channel_num, width, height = k.size()
        k = k.permute(0, 2, 3, 1).contiguous().view(batch_size, width, height, channel_num, 1, 1)
        return k

    def k_down_forward(self, x):
        k = self.key_down_conv(x)
        batch_size, channel_num, width, height = k.size()
        k = k.permute(0, 2, 3, 1).contiguous().view(batch_size, width, height, channel_num, 1, 1)
        return k

    def v_1_forward(self, x):
        v = self.value_1_conv(x)
        batch_size, channel_num, width, height = v.size()
        v = v.permute(0, 2, 3, 1).contiguous().view(batch_size, width, height, channel_num, 1, 1)
        return v

    def v_2_forward(self, x):
        v = self.value_2_conv(x)
        batch_size, channel_num, width, height = v.size()
        v = v.permute(0, 2, 3, 1).contiguous().view(batch_size, width, height, channel_num, 1, 1)
        return v
