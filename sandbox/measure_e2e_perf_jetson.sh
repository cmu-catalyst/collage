#!/bin/bash

# 1) Make sure you have right PROJ_PATH

# Currently available networks:
# resnet50, resnext50, nasnetamobile
#PROJ_PATH=~/backend-aware-graph-opt
#cd $PROJ_PATH

# CUDA_VISIBLE_DEVICES=0 --> RTX3070
# CUDA_VISIBLE_DEVICES=1 --> RTX2070
# "resnet50", "resnext50_32x4d", "bert", "nasrnn", "nasneta"
# CUDA_VISIBLE_DEVICES=0 python3 testing/measure_end_to_end.py -n resnet50 -hw jetson
# CUDA_VISIBLE_DEVICES=0 python3 testing/measure_end_to_end.py -n bert -hw jetson
# CUDA_VISIBLE_DEVICES=0 python3 testing/measure_end_to_end.py -n nasneta -hw jetson
CUDA_VISIBLE_DEVICES=0 python3 testing/measure_end_to_end.py -n mobilenet_v2 -hw jetson
#CUDA_VISIBLE_DEVICES=0 python3 testing/measure_end_to_end.py -n resnet50_3d -hw jetson
#CUDA_VISIBLE_DEVICES=0 python3 testing/measure_end_to_end.py -n resnext50_32x4d -hw jetson
#CUDA_VISIBLE_DEVICES=0 python3 testing/measure_end_to_end.py -n nasrnn -hw jetson
