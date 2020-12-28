# -*- coding: utf-8 -*-
# !@time: 2021/1/22 下午8:52
# !@author: superMC @email: 18758266469@163.com
# !@fileName: faster_rcnn_r50_attnfpn_1x_voc.py.py

_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/voc12.py',
    '../_base_/default_runtime.py'
]
model = dict(neck=dict(type='AttnFPN'), roi_head=dict(bbox_head=dict(num_classes=20)))

# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
# actual epoch = 3 * 3 = 9
lr_config = dict(policy='step', step=[3])
# runtime settings
total_epochs = 4  # actual epoch = 4 * 3 = 12
fp16 = dict(loss_scale=512.)
