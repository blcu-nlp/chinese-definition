#!/bin/bash
set -x
set -e

source ../paths.sh
fairseqpy=$

# download embeddings if necessary
embed_path=$/data/private/fqn/workspace/Transformer/fairseq/word2vec/chinesegigawordv5.jieba.skipngram.300d.txt

seed=1
data_bin_dir=
out_dir=
mkdir -p $out_dir

python_path=
CUDA_VISIBLE_DEVICES= python $fairseqpy/train.py
    --save-dir $out_dir\
    --encoder-embed-dim 500\
    --encoder-embed-path $embed_path\
    --decoder-embed-dim 500\
    --decoder-embed-path $embed_path\
    --decoder-out-embed-dim 500\
    --dropout 0.2\
    --clip-norm 0.1\
    --lr 0.25\
    --min-lr 1e-4\ 
    --encoder-layers '[(1024,3)] * 7'\
    --decoder-layers '[(1024,3)] * 7'\ 
    --momentum 0.99\ 
    --max-epoch 100\ 
    --batch-size 32\ 
    --no-progress-bar\ 
    --seed $seed $data_bin_dir


