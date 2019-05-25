#! /usr/bin/env bash
path=processed/results
mkdir -p $path

python ./scripts/cut_and_split.py \
    ./processed/dataset.txt.new \
    $path