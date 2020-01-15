# -*- coding: utf-8 -*-
import sys
from collections import defaultdict


def read_sememe_data(path):
    word_sememe_definition = defaultdict(list)
    with open(path) as fr:
        for line in fr:
            line = line.strip().split(' ||| ')
            if len(line) != 5:
                continue
            word = line[0]
            sememe = line[-2].split(' ')
            sememe = [s.split('|')[1] for s in sememe]
            sememe = ' '.join(sememe)
            if len(sememe) == 0:
                sememe = word
            if '；' in line[-1]:
                definitions = line[-1].split('；')
                for d in definitions:
                    word_sememe_definition[word].append([sememe, d])
            else:
                word_sememe_definition[word].append([sememe, line[-1]])
    return word_sememe_definition


def read_definition_data(path):
    word_definition = defaultdict(list)
    with open(path) as fr:
        for line in fr:
            line = line.strip().split(' ||| ')
            word = line[0]
            if '；' in line[-1]:
                definitions = line[-1].split('；')
                for d in definitions:
                    word_definition[word].append([word, d])
            else:
                word_definition[word].append([word, line[-1]])
    return word_definition


def merge(word_sememe_definition, word_definition):
    dataset = {}
    for word in word_definition:
        if word in word_sememe_definition:
            dataset[word] = word_sememe_definition[word]
            definitions = [d[1] for d in dataset[word]]
            for [s, d] in word_definition[word]:
                if d not in definitions:
                    dataset[word].append([s, d])
        else:
            dataset[word] = word_definition[word]
    return dataset


def write2file(dataset, path):
    with open(path, 'w') as fw:
        for word in dataset:
            for sememe, definition in dataset[word]:
                fw.write('{} ||| {} ||| {}\n'.format(word, sememe, definition))


def main(argv=None):
    if argv is None:
        argv = sys.argv
        if len(argv) != 4:
            raise ValueError("Args Num Not Match.")
    word_sememe_definition = read_sememe_data(argv[1])
    word_definition = read_definition_data(argv[2])
    dataset = merge(word_sememe_definition, word_definition)
    write2file(dataset, argv[3])
    return 0


if __name__ == '__main__':
    sys.exit(main())
