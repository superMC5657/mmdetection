# -*- coding: utf-8 -*-
# !@time: 2021/1/22 下午8:52
# !@author: superMC @email: 18758266469@163.com
# !@fileName: faster_rcnn_r50_attnfpn_1x_coco.py.py

_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/default_runtime.py',
]
model = dict(neck=dict(type='AttnFPN'))
fp16 = dict(loss_scale=512.)

data = dict(
    samples_per_gpu=3,
    workers_per_gpu=3)
optimizer = dict(
    type='AdamW',
    lr=0.001,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.1, decay_mult=1.0)}))
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
# learning policy
lr_config = dict(policy='step', step=[100])
total_epochs = 15
