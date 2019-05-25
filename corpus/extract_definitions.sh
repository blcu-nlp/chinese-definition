#! /usr/bin/env bash

python ./scripts/extract_definitions.py\
       ./xls\
       ./processed/sorted_words.txt\
       ./processed/word_definitions.txt
