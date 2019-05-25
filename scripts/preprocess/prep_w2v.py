# -*- coding: utf-8 -*-
from gensim.models import word2vec
import json
import numpy as np
import sys


def main(argv=None):
    if argv is None:
        argv = sys.argv
        if len(argv) != 4:
            raise ValueError("Args Num Not Match.")
    print('Making Matrix...')
    with open(argv[1]) as fr:
        word2idx = json.loads(fr.read())
    print('Loading Model...')
    model = word2vec.Word2VecKeyedVectors.load_word2vec_format(
        argv[2], binary=True)
    matrix = np.zeros((len(word2idx) + 1, model.vector_size))
    for word in word2idx:
        try:
            matrix[word2idx[word]] = model.word_vec(word)
        except KeyError:
            matrix[word2idx[word]] = model.word_vec('UNK'.encode('utf-8'))
    np.save(argv[3], matrix)


if __name__ == '__main__':
    sys.exit(main())
