#!/usr/bin/env python
python train.py -path /media/awmagsam/HDD/Datasets/KITTI \
                -batch_size 6 \
                -n_iters 20000 \
                -test_freq 250 \
                -save_freq 1000 \
                -prefix resnet50_v2 \
                -lr 0.0001 \
                -device 0