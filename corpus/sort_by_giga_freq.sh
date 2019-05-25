#! /usr/bin/env bash

python scripts/sort_by_giga_freq.py\
    utils/gigawords_freq_sun.txt\
    processed/same_words.txt\
    processed/word_sememes.txt\
    processed/sorted_words.txt
