# -*- coding: utf-8 -*-
# !@time: 2020/12/16 21 41
# !@author: superMC @email: 18758266469@163.com
# !@fileName: kda_loss.py
from abc import ABC

import torch
from torch import nn

try:
    from ..multi_focal_loss import CrossEntropyWithLogitsLoss
except:
    from mmdet.models.losses.multi_focal_loss import CrossEntropyWithLogitsLoss


class MultiHeadSelfAttentionLoss(nn.Module, ABC):
    def __init__(self, alpha=0.1):
        super().__init__()
        self.loss_func = CrossEntropyWithLogitsLoss(alpha=alpha)

    def forward(self, f1, f2):
        batch_size, multi_head, channel_num, channel_num = f1.size()
        f1 = f1.view(-1, channel_num)
        f2 = f2.view(-1, channel_num)
        loss = self.loss_func(f1, f2)
        return loss


class RelationFeatureLoss(nn.Module, ABC):
    def __init__(self, gamma=0.1):
        super().__init__()
        self.gamma = gamma

    def forward(self, f1, f2):
        pass


if __name__ == '__main__':
    from torch.nn import functional as F

    cal = MultiHeadSelfAttentionLoss(alpha=1)
    f1 = torch.rand((1, 2, 4, 4))
    f2 = torch.ones((1, 2, 4, 4))
    loss = cal(f1, f2)
    print(loss)

    f1 = F.softmax(f1, dim=-1)
    f2 = F.softmax(f2, dim=-1)

    loss = cal(f1, f2)
    print(loss)
