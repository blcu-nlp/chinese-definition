#!/usr/bin/env bash

# references=("../data/commondefs/test.txt" "../data/commondefs/models/inter/test_gcide.txt" "../data/commondefs/models/inter/test_wordnet.txt")
r="../data/commondefs/test.txt"
#for r in $references
# do
	echo $r
    echo $1"/test_samples"$2"_"$3".txt"$4
    python bleu.py $r  $1"/test_samples"$2"_"$3".txt"$4
# done
