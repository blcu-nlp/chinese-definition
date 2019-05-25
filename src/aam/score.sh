#! /usr/bin/env bash
CUDA_VISIBLE_DEVICES=0,1
model_path=../models/adaptive/$1
epoch=$2

python ../scripts/eval/rerank.py\
    $model_path/greedy_output.txt\
    ../../data/chinesegigawordv5.lm\
    ../../data/function_words.txt\
    $model_path/greedy_output_rank.txt

python score.py\
    --gen_file_path $model_path/greedy_output_rank.txt\
    --pretrained $model_path/adaptive-$2.pkl\
    --output_path $model_path/score_greedy_output_rank.txt

python ../scipts/eval/rerank2.py\
    $model_path/greedy_output_rank.txt\
    $model_path/score_greedy_output_rank.txt\
    ../../data/function_words.txt\
    $model_path/greedy_output_rank2.txt.top

python ./metrics/bleu.py\
    ../../data/test.txt\
    $model_path/greedy_output_rank2.txt.top
