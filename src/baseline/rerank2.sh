#!/usr/bin/env bash

DATA=data/commondefs/models/m2
for t in 1 .5 .25 .1 .05
do
	echo $DATA"/valid_samples"$t"_rank.txt"
	python rerank2.py $DATA"/valid_samples"$t"_rank.txt" $DATA"/score_valid_samples"$t"_rank.txt"  data/function_words.txt $DATA"/valid_samples"$t"_rank2.txt.top"
	echo $DATA"/test_samples"$t"_rank.txt"
	python rerank2.py $DATA"/test_samples"$t"_rank.txt" $DATA"/score_test_samples"$t"_rank.txt"  data/function_words.txt $DATA"/test_samples"$t"_rank2.txt.top"
done
