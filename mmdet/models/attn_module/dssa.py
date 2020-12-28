# -*- coding: utf-8 -*-
# !@time: 2020/12/24 下午3:40
# !@author: superMC @email: 18758266469@163.com
# !@fileName: dssa.py
from abc import ABC
from torchvision.ops.roi_pool import RoIPool
import torch
from torch import nn


class DiffScaleSpatialAttention(nn.Module, ABC):
    def __init__(self, size=10):
        super(DiffScaleSpatialAttention, self).__init__()
        self.roi_pool = RoIPool(output_size=size, spatial_scale=1)

