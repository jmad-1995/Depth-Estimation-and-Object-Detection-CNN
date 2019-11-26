#!/usr/bin/env python
python train.py -path /media/awmagsam/HDD/Datasets/KITTI \
                -batch_size 2 \
                -n_iters 20000 \
                -test_freq 200 \
                -save_freq 1000 \
                -prefix resnet50_v1 \
                -lr 0.0001 \
                -device 0