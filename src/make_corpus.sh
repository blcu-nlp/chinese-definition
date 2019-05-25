#! /usr/bin/env bash
cd ../corpus
./count_words.sh
./extract_sememes.sh
./extract_same_words.sh
./sort_by_giga_freq.sh
./extract_definitions.sh
./wsd.sh
./make_final_dataset.sh
./del_oov.sh
./cut_and_split.sh
cp ./processed/results/train.txt ../data/
cp ./processed/results/valid.txt ../data/
cp ./processed/results/test.txt ../data/