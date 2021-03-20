# -*- coding: utf-8 -*-
# !@time: 2020/12/23 下午9:20
# !@author: superMC @email: 18758266469@163.com
# !@fileName: demo.py
import time

from mmdet.models.necks.attn_necks.attn_pafpn import AttnPAFPN, AttnPAFPNV2
from mmdet.models.necks.attn_necks.attn_fpn import AttnFPN, AttnFPNV2, AttnFPNV3

if __name__ == '__main__':
    import torch

    # self = AttnResNet(depth=34)
    # self.cuda()
    # inputs = torch.ones(16, 3, 640, 640).cuda()
    # level_outputs, attns = self.forward(inputs)
    # # level_outputs = self.forward(inputs)
    # for level_out in level_outputs:
    #     print(tuple(level_out.shape))
    # for attn in attns:
    #     print(tuple(attn.shape))
    in_channels = [2, 3, 5]
    scales = [68, 34, 17]
    inputs = [torch.rand(1, c, s, s).cuda() for c, s in zip(in_channels, scales)]
    self = AttnFPNV3(in_channels, 256, len(in_channels) + 3).cuda().eval()
    #
    for i in range(100):
        start = time.time()

        outputs = self(inputs)
        print(time.time() - start)

    # torch.save(self, 'fpn.pth')
    # x1 = torch.randn((2, 256))
    # x2 = torch.randn((256, 2))
    # y = torch.matmul(x1, x2)
    # print(y)

    # x1 = torch.rand(1, 250, 12, 12)
    # x2 = torch.rand(1, 250, 12, 12)
    # print(x1, x2)
    # x3 = x1 * x2
    # print(x3)
    # print(x3.size())


