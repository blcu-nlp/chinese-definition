#! /usr/bin/env bash
cd ../scripts/make_dataset
./count_words.sh
./extract_sememes.sh
./extract_same_words.sh
./sort_by_giga_freq.sh
./extract_definitions.sh
./wsd.sh
./make_final_dataset.sh
./del_oov.sh
./cut_and_split.sh
cp ./processed/results/train.txt ../dataset/
cp ./processed/results/valid.txt ../dataset/
cp ./processed/results/test.txt ../dataset/
