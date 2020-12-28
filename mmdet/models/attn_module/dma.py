# -*- coding: utf-8 -*-
# !@time: 2020/12/11 下午3:36
# !@author: superMC @email: 18758266469@163.com
# !@fileName: dma.py
# dual MultiHeadSelfAttention
from abc import ABC
from math import ceil

import torch

from torch import nn, matmul
from torch.nn import functional as F


class MultiHeadSpatialSelfAttention(nn.Module, ABC):
    def __init__(self, in_planes, hidden_state=16, multi_head=4, alpha=0.5, fuse=False):
        super().__init__()
        hidden_state = multi_head * hidden_state
        self.multi_head = multi_head
        self.hidden_state = hidden_state

        self.query_conv = nn.Conv2d(in_channels=in_planes, out_channels=hidden_state, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_planes, out_channels=hidden_state, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_planes, out_channels=hidden_state, kernel_size=1)
        self.conv = nn.Conv2d(in_channels=hidden_state, out_channels=in_planes, kernel_size=1)
        self.bn = nn.BatchNorm2d(in_planes)
        if fuse and multi_head > 1:
            self.fuse = nn.Conv2d(in_channels=multi_head, out_channels=multi_head, kernel_size=1, bias=False)
        else:
            self.fuse = None
        self.attention = ScaledDotProductAttention(temperature=hidden_state ** alpha, fuse=self.fuse)

    def forward(self, x):
        batch_size, channel_num, width, height = x.size()
        proj_query = self.query_conv(x).view(batch_size, self.multi_head, -1, width * height).transpose(2,
                                                                                                        3).contiguous()
        proj_key = self.key_conv(x).view(batch_size, self.multi_head, -1, width * height)
        out = self.value_conv(x).view(batch_size, self.multi_head, -1, width * height).transpose(2,
                                                                                                 3).contiguous()

        out, attn = self.attention(proj_query, proj_key, out)
        out = out.transpose(2, 3).contiguous().view(batch_size, -1, width, height)
        out = self.conv(out)
        out = self.bn(out)

        return out, attn


class MultiHeadChannelSelfAttention(nn.Module, ABC):
    def __init__(self, in_planes, hidden_state=16, multi_head=4, beta=0.5, fuse=False):
        super(MultiHeadChannelSelfAttention, self).__init__()
        hidden_state = multi_head * hidden_state
        self.multi_head = multi_head
        self.hidden_state = hidden_state

        self.query_conv = nn.Conv2d(in_channels=in_planes, out_channels=hidden_state, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_planes, out_channels=hidden_state, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_planes, out_channels=hidden_state, kernel_size=1)

        self.conv = nn.Conv2d(in_channels=hidden_state, out_channels=in_planes, kernel_size=1)
        self.bn = nn.BatchNorm2d(in_planes)
        if fuse and multi_head > 1:
            self.fuse = nn.Conv2d(in_channels=multi_head, out_channels=multi_head, kernel_size=1)
        else:
            self.fuse = None

        self.attention = ScaledDotProductAttention(temperature=hidden_state ** beta, fuse=self.fuse)

    def forward(self, x):
        batch_size, channel_num, width, height = x.size()
        proj_query = self.query_conv(x).view(batch_size, self.multi_head, -1, width * height)
        proj_key = self.key_conv(x).view(batch_size, self.multi_head, -1, width * height).transpose(2, 3).contiguous()
        proj_value = self.value_conv(x).view(batch_size, self.multi_head, -1, width * height)

        out, attn = self.attention(proj_query, proj_key, proj_value)
        out = out.view(batch_size, -1, width, height)
        out = self.conv(out)
        out = self.bn(out)

        return out, attn


class MultiHeadSpatialSelfAttentionWithDownSampler(MultiHeadSpatialSelfAttention, ABC):
    def __init__(self, in_planes, hidden_state=16, multi_head=4, alpha=0.5, fuse=False, ratio=1):
        super().__init__(in_planes, hidden_state, multi_head, alpha, fuse)
        self.downSample = nn.AvgPool2d(ratio, stride=ratio)
        self.upSample = nn.Upsample(scale_factor=ratio, mode='nearest')

    def forward(self, x):
        x = self.downSample(x)
        out, attn = super(MultiHeadSpatialSelfAttentionWithDownSampler, self).forward(x)
        out = self.upSample(out)
        return out, attn


class MultiHeadSpatialSelfAttentionWithDownSamplerV2(MultiHeadSpatialSelfAttention, ABC):
    def __init__(self, in_planes, hidden_state=16, multi_head=4, alpha=0.5, fuse=False, ratio=1):
        super().__init__(in_planes, hidden_state, multi_head, alpha, fuse)
        self.ratio = ratio

    def forward(self, x):
        batch_size, channel_num, width, height = x.size()
        pool_size = ceil(width / self.ratio)
        x = F.adaptive_avg_pool2d(input=x, output_size=(pool_size, pool_size))
        out, attn = super().forward(x)
        out = F.upsample(x, size=width, mode='nearest')
        return out, attn


class ScaledDotProductAttention(nn.Module, ABC):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature, attn_dropout=0.1, fuse=None):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout, inplace=False)
        self.softmax = nn.Softmax(dim=-1)
        self.fuse = fuse
    def forward(self, q, k, v):
        attn = matmul(q / self.temperature, k)
        if self.fuse:
            attn = self.fuse(attn)
        attn = self.dropout(self.softmax(attn))
        v = matmul(attn, v)
        return v, attn


if __name__ == '__main__':
    from torchsummaryX import summary
    from thop import profile
    from fvcore.nn import flop_count

    x = torch.rand((1, 32, 20, 20))
    mca = MultiHeadChannelSelfAttention(32, 16, 4, fuse=False)
    msa = MultiHeadSpatialSelfAttention(32, 16, 4, fuse=False)
    # flops, skip = flop_count(msa, (x,))
    # print("%s|%s" % (flops, skip))
    # out, attn = msa(x)
    # print(out.size(), attn.size())

    # summary(mca, (32, 20, 20), device='cpu')
    # summary(msa, (32, 20, 20), device='cpu')

    # summary(mca, x)
    # summary(msa, x)
    flops, params = profile(mca, inputs=(x,))
    print(flops, params)
    flops, params = profile(msa, inputs=(x,))
    print(flops, params)
