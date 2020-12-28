# -*- coding: utf-8 -*-
# !@time: 2020/12/20 17 58
# !@author: superMC @email: 18758266469@163.com
# !@fileName: loss_utils.py
from abc import ABC

import torch
from torch import nn


class MultiFocalLoss(nn.Module, ABC):

    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(MultiFocalLoss, self).__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, predict, target):
        epsilon = 1e-10
        predict = predict.log()
        ce_loss = -(target * predict).sum(1) + epsilon

        focal_loss = self.alpha * torch.pow((1 - ce_loss), self.gamma) * ce_loss.log()
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss


class CrossEntropyWithLogitsLoss(nn.Module, ABC):
    def __init__(self, alpha=1, reduction='mean'):

        super().__init__()
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, predict, target):
        predict = predict.log()

        ce_loss = -(target * predict).sum(1) * self.alpha
        if self.reduction == 'mean':
            return ce_loss.mean()
        elif self.reduction == 'sum':
            return ce_loss.sum()
        else:  # 'none'
            return ce_loss


if __name__ == '__main__':
    import torch.nn.functional as F

    mfl = MultiFocalLoss(gamma=2)
    predict = torch.randn((3, 4))
    target = torch.randn((3, 4))
    predict = F.softmax(predict, dim=1)
    target = F.softmax(target, dim=1, )
    ce_loss = mfl(predict, target)
    print(ce_loss)
