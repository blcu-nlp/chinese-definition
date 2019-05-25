#! /usr/bin/env bash
model_path="../models/adaptive/$1"
epoch=$2
CUDA_VISIBLE_DEVICES=0,1 python src/inference.py\
    --pretrained $model_path/adaptive-$epoch.pkl\
    --output_path $model_path/greedy_output.txt\
    --seed 1024\
    --beam_size 1\
    --test_size 50
