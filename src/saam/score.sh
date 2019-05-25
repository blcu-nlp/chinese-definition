#! /usr/bin/env bash
model_path=./checkpoints/$1

set -x
set -e

for i in `seq 240 266`
do
    for x in 'valid' 'test'
    do
        echo -n "epoch $i" >> result-bleu.txt
        echo -n " $x " >> result-bleu.txt

        GEN="$model_path/${x}_${i}.txt"
        
        grep ^S $GEN | cut -f2- > $model_path/gen_src.txt
        grep ^H $GEN | cut -f3- > $model_path/gen_tgt.txt

        python ./metrics/concator.py\
            $model_path/gen_src.txt \
            $model_path/gen_tgt.txt \
            $model_path/result.txt

        python ./metrics/rerank.py\
            $model_path/result.txt\
            ./data/lm.bin\
            ./data/function_words.txt\
            $model_path/result.txt.rank

        python ./metrics/bleu.py\
            ./data/${x}.txt\
            $model_path/result.txt.rank.top >> result-bleu-warmup.txt
    done
done

