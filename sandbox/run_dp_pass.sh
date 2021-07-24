#!/bin/bash
# CUDA_VISIBLE_DEVICES=0 --> RTX3070
# CUDA_VISIBLE_DEVICES=1 --> RTX2070
# "resnet50", "resnext50_32x4d", "bert", "nasrnn", "nasneta", "conv2d"
#CUDA_VISIBLE_DEVICES=1 python3 testing/test_relay_build.py -n resnet50 -hw rtx2070
#CUDA_VISIBLE_DEVICES=1 python3 testing/test_relay_build.py -n resnext50_32x4d -hw rtx2070
#CUDA_VISIBLE_DEVICES=1 python3 testing/test_relay_build.py -n nasrnn -hw rtx2070
#CUDA_VISIBLE_DEVICES=1 python3 testing/test_relay_build.py -n bert -hw rtx2070
#CUDA_VISIBLE_DEVICES=1 python3 testing/test_relay_build.py -n nasneta -hw rtx2070
CUDA_VISIBLE_DEVICES=0 python3 testing/test_relay_build.py -n resnet50_3d -hw jetson2
#CUDA_VISIBLE_DEVICES=0 python3 testing/test_relay_build.py -n mobilenet_v2 -hw rtx3070

# Relay examples
#CUDA_VISIBLE_DEVICES=0 python3 testing/test_relay_build.py -n annotate_test -hw rtx2070
#CUDA_VISIBLE_DEVICES=1 python3 testing/test_relay_build.py -n conv2d -hw rtx2070
#CUDA_VISIBLE_DEVICES=1 python3 testing/test_relay_build.py -n conv2d+relu -hw rtx2070
#CUDA_VISIBLE_DEVICES=1 python3 testing/test_relay_build.py -n conv2d+relu_x2 -hw rtx2070


