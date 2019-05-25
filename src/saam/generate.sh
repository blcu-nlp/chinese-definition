model_path=checkpoints/$1

for i in `seq 240 266`
do
    echo $i
    CUDA_VISIBLE_DEVICES=0 python -u generate.py data/bin \
        --path $model_path/checkpoint$i.pt \
        --gen-subset valid \
        --batch-size 256 \
        --beam 1\
        --nbest 1 \
        --print-alignment\
        --no-progress-bar | tee $model_path/valid_$i.txt

    CUDA_VISIBLE_DEVICES=0 python -u generate.py data/bin \
        --path $model_path/checkpoint$i.pt \
        --batch-size 256 \
        --beam 1\
        --nbest 1 \
        --print-alignment\
        --no-progress-bar | tee $model_path/test_$i.txt
done
