model_path=checkpoints/$1
echo "model path: $model_path"
mkdir -p $model_path
embed_path=./data/word2vec/selected_embeddings.txt
CUDA_VISIBLE_DEVICES=2,3 python -u train.py data/bin\
    --arch transformer\
    --optimizer adam\
    --adam-betas '(0.9, 0.98)'\
    --clip-norm 0.1\
    --lr-scheduler inverse_sqrt\
    --lr 1e-4\
    --min-lr 1e-09\
    --encoder-attention-heads 5\
    --encoder-embed-dim 300\
    --encoder-embed-path $embed_path\
    --decoder-attention-heads 5\
    --decoder-embed-dim 300\
    --decoder-embed-path $embed_path\
    --dropout 0.2\
    --criterion label_smoothed_cross_entropy\
    --label-smoothing 0.1\
    --batch-size 128\
    --update-freq 16\
    --encoder-learned-pos \
    --decoder-learned-pos \
    --max-epoch 500\
    --save-dir $model_path\
    --seed 1024\
    --weight-decay 1e-5\
    --warmup-init-lr 1e-07\
    --warmup-updates 2000\
    --ddp-backend=no_c10d \
    2>&1 | tee -a $model_path/training_log.txt
    #--encoder-ffn-embed-dim 600 \
    #--decoder-ffn-embed-dim 600 \
    #--no-epoch-checkpoints\
    #--max-tokens 44190\
    #--criterion label_smoothed_cross_entropy\
    #--label-smoothing 0.1\
    #--encoder-layers 7\
    #--decoder-layers 7\
