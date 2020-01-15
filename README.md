# Chinese Definition Modeling

This repository is connected with Chinese Definition Modeling task.

The src directory consists of three models:

- baseline
- Adaptive-Attention Model
- Self- and Adaptive-Attention Model

Paper Link: https://arxiv.org/abs/1905.06512

Contact: cunliang.kong@outlook.com

## Dataset Construction

### Requirements

- python (3.6)
- xlrd (1.1.0)
- jieba (0.39)
- progressbar2 (3.38.0)

### Construction

The dataset construction procedure follows the `README.md`file in the `scripts/make_dataset` directory.

We have also written an integreted script `make_dataset.sh` in the directory of `src`. 

```bash
cd src
chmod +x make_dataset.sh
./make_dataset.sh
```

The CWN dataset we used in the experiments is in the `dataset/cwn` directory.

## Baseline

The baseline model is based on [Websail-NU/torch-defseq](https://github.com/websail-nu/torch-defseq), and detailed instruction can be found there.

## Adaptive-Attention Model

The Adaptive-Attention model is in the directory of `src/aam`, and can run as follows:

- Requirements

  - python (2.7)
  - pytorch (0.3.1)
  - numpy (1.14.5)
  - gensim (3.5.0)
  - [kenlm](https://github.com/kpu/kenlm/)

- Preprocess

    The preprocess procedure is written in the script of `preprocess.sh`. During preprocessing, we used pretrained Chinese word embeddings, which is trained on the [Chinese Gigaword Corpus](https://catalog.ldc.upenn.edu/LDC2011T13). [Jieba](https://github.com/fxsjy/jieba) Chinese segmentation tool is employed. The binarized word2vec file is named `gigaword_300d_jieba.bin` placed in the directory of `data`.

    ```bash
    cd src/adaptive
    ./preprocess.sh
    ```

- Training & Inference

    You can use following commands to train and inference. Also, we've uploaded the `training_lot.txt` of the best model in the directory of `models/adaptive/best`.

    ```bash
    ./train.sh best #using the best parameters to train a model
    ./inference.sh best 22 #22 denotes the best epoch
    ```

- Scoring
  - A `function_words.txt` is needed in the `data` directory, we've extracted one from the HowNet when making the dataset
  - A `chinesegigawordv5.lm` Chinese language model is needed in the `data` directory, any arpa format language model will do
  - Then you can use the following script to compute the score of BLEU
  ```bash
  ./score.sh best 21 #21 denotes the best epoch
  ```

## Self- and Adaptive-Attention Model

The Self- and Adaptive-Attention Model is in the directory of `src/saam`.  The instruction of this model is as follows:

- Requirements and Installation

  - python (3.6)
  - pytorch (0.4.1)
  - use following commands to install other requirements

  ```bash
  cd src/self-attention
  pip install -r requirements.txt
  ```

- Preprocess

  The preprocessing scripts is used to convert text files into binarized data.

  ```bash
  ./preprocess.sh
  ```

- Train & Generate

  We use fixed pre-trained word embeddings as the adaptive attention model. The word embedding is in the directory of `data` and named `chinesegigawordv5.jieba.skipngram.300d.txt`. We uploaded a demo word embedding file which contains only 100 lines.

  The model can be trained and employed using following commands:

  ```bash
  ./train.sh best #best is name of the model
  ./generate.sh best
  ```

  Parameters used for training is written in the `train.sh` script
