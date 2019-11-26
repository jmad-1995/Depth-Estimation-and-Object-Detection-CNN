#!/usr/bin/env python
python train.py -path /media/awmagsam/HDD/Datasets/KITTI \
                -batch_size 4 \
                -n_iters 10000 \
                -test_freq 100 \
                -save_freq 500 \
                -prefix resnet50_v1 \
                -lr 0.0001 \
                -device 0