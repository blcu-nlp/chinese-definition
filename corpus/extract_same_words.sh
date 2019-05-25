#! /usr/bin/env bash

cat ./utils/HowNet\ Chinese\ Word\ List.txt \
    | awk '{print $1}' > ./processed/hownet_word_list.txt

python ./scripts/extract_same_words.py\
    ./processed/ccd_word_list.txt\
    ./processed/hownet_word_list.txt\
    ./processed/same_words.txt
