# -*- coding: utf-8 -*-
import sys
import os
# from pyhanlp import HanLP
import jieba
from collections import defaultdict
from progressbar import progressbar
import random


def read_dataset(path):
    dataset = defaultdict(list)
    with open(path) as fr:
        for line in fr:
            line = line.strip().split(' ||| ')
            if len(line) != 3:
                continue
            word = line[0]
            sememe = line[1]
            if '；' in line[2]:
                definitions = line[2].split('；')
                for d in definitions:
                    dataset[word].append([sememe, d])
            else:
                d = line[2]
                dataset[word].append([sememe, d])
    return dataset


def cut_words(dataset):
    dataset_cut = defaultdict(list)
    print('cutting...')
    for word in progressbar(dataset):
        for s, d in dataset[word]:
            # cut_d = HanLP.segment(d)
            # dataset_cut[word].append(' '.join([c.word for c in cut_d]))
            cut_d = jieba.lcut(d)
            dataset_cut[word].append([s, ' '.join(cut_d)])
    return dataset_cut


def split_dataset(dataset_cut, path):
    nb_all = len(dataset_cut.keys())
    nb_train = round(9 / 10 * nb_all)
    nb_valid = round(0.5 / 10 * nb_all)
    all_words = list(dataset_cut.keys())
    random.shuffle(all_words)
    train_words = all_words[:nb_train]
    valid_words = all_words[nb_train:nb_train + nb_valid]
    test_words = all_words[nb_train + nb_valid:]
    train_dataset = defaultdict(list)
    valid_dataset = defaultdict(list)
    test_dataset = defaultdict(list)
    for w in train_words:
        train_dataset[w] = dataset_cut[w]
    for w in valid_words:
        valid_dataset[w] = dataset_cut[w]
    for w in test_words:
        test_dataset[w] = dataset_cut[w]

    def write_datasets(corpus, path):
        with open(path, 'w') as fw:
            for word in corpus:
                for sememe, definition in corpus[word]:
                    fw.write('{} ||| {} ||| {}\n'.format(
                        word, sememe, definition))

    write_datasets(train_dataset, os.path.join(path, 'train.txt'))
    write_datasets(valid_dataset, os.path.join(path, 'valid.txt'))
    write_datasets(test_dataset, os.path.join(path, 'test.txt'))
    return train_dataset, valid_dataset, test_dataset


def make_shortlist(test, path):
    shortlist_test = []
    for word in test:
        for sememe, _ in test[word]:
            if (word, sememe) not in shortlist_test:
                shortlist_test.append((word, sememe))
    with open(os.path.join(path, 'shortlist_test.txt'), 'w') as fw:
        for word, sememe in shortlist_test:
            fw.write('{} ||| {}\n'.format(word, sememe))


def main(argv=None):
    if argv is None:
        argv = sys.argv
        if len(argv) != 3:
            raise ValueError("Args num not match.")
    dataset = read_dataset(argv[1])
    dataset = cut_words(dataset)
    train, valid, test = split_dataset(dataset, argv[2])
    make_shortlist(test, argv[2])
    return 0


if __name__ == '__main__':
    sys.exit(main())
