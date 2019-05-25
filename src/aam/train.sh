#! /usr/bin/env bash
model_path="../../models/adaptive/$1"
mkdir -p $model_path
CUDA_VISIBLE_DEVICES=0,1 python train.py\
    --model_path $model_path\
    --seed 1024\
    --alpha 0.9\
    --beta 0.999\
    --learning_rate 1e-3\
    --embed_size 300\
    --hidden_size 512\
    --batch_size 128\
    --clip 1\
    --lr_decay 20\
    --num_epochs 100\
    --learning_rate_decay_every 50\
    #2>&1 | tee $model_path/training_log.txt
    #--pretrained $model_path/adaptive-14.pkl\
