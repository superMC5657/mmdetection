# -*- coding: utf-8 -*-
# !@time: 2021/1/24 下午9:30
# !@author: superMC @email: 18758266469@163.com
# !@fileName: faster_rcnn_r50_attnfpnv2_1x_coco.py

_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_1x.py'
]
model = dict(neck=dict(type='AttnFPNV2'))
fp16 = dict(loss_scale=512.)
data = dict(
    samples_per_gpu=3,
    workers_per_gpu=3)

