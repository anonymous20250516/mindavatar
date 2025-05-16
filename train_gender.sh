#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python train_gender.py --sub 1 --train_num 3000 --epochs 100 --batch_size 64 --lr 1e-4 \
        --use_vc --weight_decay 0.05 \
        --ckpt_dir ./results/gender

