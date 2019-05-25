#! /usr/bin/env bash

python ./scripts/get_hypernym.py\
    ./xls\
    ./processed/dataset.txt.new\
    ./processed/bag_of_hypernyms.txt
