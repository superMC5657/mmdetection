# -*- coding: utf-8 -*-
# !@time: 2020/12/23 下午9:20
# !@author: superMC @email: 18758266469@163.com
# !@fileName: demo.py
from mmdet.models import ResNet

if __name__ == '__main__':
    from mmdet.models.backbones.attn_backbones.attn_resnet import AttnResNet
    import torch

    self = AttnResNet(depth=34)
    self.cuda()
    self.eval()
    inputs = torch.rand(32, 3, 320, 320).cuda()
    level_outputs, attns = self.forward(inputs)
    # level_outputs = self.forward(inputs)
    for level_out in level_outputs:
        print(tuple(level_out.shape))
    for attn in attns:
        print(tuple(attn.shape))
