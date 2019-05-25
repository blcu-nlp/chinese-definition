TEXT=./data/processed
python preprocess.py\
    --srcdict $TEXT/dict.txt\
    --tgtdict $TEXT/dict.txt\
    --source-lang src\
    --target-lang tgt\
    --trainpref $TEXT/train\
    --validpref $TEXT/valid\
    --testpref $TEXT/test\
    --destdir data/bin\
    --thresholdtgt 0\
    --thresholdsrc 0\
    --output-format binary
