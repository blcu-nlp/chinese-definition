TEXT=examples/translation/trans_def/trans_def.src-tgt
python learn_joint_bpe_and_vocab.py\
    --input $TEXT/train.src $TEXT/train.tgt\
    --symbols
    --output
    --write-vocabulary

