#! /usr/bin/env bash
python ../../scripts/preprocess/prep_datasets.py
python ../../scripts/preprocess/prep_w2v.py\
    ../../data/processed/sememe2idx.json\
    ../../data/gigaword_300d_jieba_unk.bin\
    ../../data/processed/sememe_matrix.npy
python ../../scripts/preprocess/prep_w2v.py\
    ../../data/processed/word2idx.json\
    ../../data/gigaword_300d_jieba_unk.bin\
    ../../data/processed/word_matrix.npy
