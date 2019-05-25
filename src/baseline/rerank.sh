#!/usr/bin/env bash

DATA=data/commondefs/models/tmp_exp
for t in 1 .5 .25 .1 .05
    do
        echo $DATA"/valid_samples"$t"_gen.txt"
        ipython rerank.py $DATA"/valid_samples"$t"_gen.txt"\
        "./data/commondefs/models/ngram/models/ngram/lm.arpa" data/function_words.txt\
        $DATA"/valid_samples"$t"_rank_lm.txt"

        echo $DATA"/test_samples"$t"_gen.txt"
        ipython rerank.py $DATA"/test_samples"$t"_gen.txt"\
        "./data/commondefs/models/ngram/models/ngram/lm.arpa" data/function_words.txt\
        $DATA"/test_samples"$t"_rank_lm.txt"
    done

#ipython rerank.py "$DATA/py_gen.txt" data/commondefs/models/ngram/lm.arpa data/function_words.txt "$DATA/test_rank.txt"
