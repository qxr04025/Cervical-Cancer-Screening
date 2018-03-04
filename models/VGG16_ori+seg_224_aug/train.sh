#!/usr/bin/env sh
set -e

/home/qinxiaoran/project/caffe/build/tools/caffe train \
    --solver=solver.prototxt  --weights=../pre_trained/VGG_ILSVRC_16_layers.caffemodel --gpu=2,3 $@
