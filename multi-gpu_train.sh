GPUS=2
PORT=23456
CONFIG=configs/attnfpn/faster_rcnn_r50_attnfpnv3_1x_coco.py
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    tools/train.py $CONFIG --launcher pytorch