# -*- coding: utf-8 -*-
import os
import sys
from collections import defaultdict

import xlrd


def read_ccd(path):
    offset2word = {}
    word2offset = defaultdict(list)
    ccd_file_format = 'CCD_Data_{}.xls'
    pos = ['Noun', 'Verb']
    for p in pos:
        ccd_file = ccd_file_format.format(p)
        print(ccd_file)
        xls_file = os.path.join(path, ccd_file)
        data = xlrd.open_workbook(xls_file)
        table = data.sheets()[0]
        nrows = table.nrows
        for i in range(1, nrows):
            line = table.row_values(i)
            words = line[4].split(' ')
            hypernyms = []
            for i in range(0, len(line[10]), 8):
                hypernyms.append(line[10][i:i + 8])

            offset2word[line[0]] = {
                'words': words,
                'pos': line[1],
                'hypernyms': hypernyms
            }
            for w in words:
                word_info = {
                    'offset': line[0],
                    'pos': line[1],
                    'hypernyms': hypernyms
                }
                if word_info not in word2offset[w]:
                    word2offset[w].append(word_info)
    return offset2word, word2offset


def read_words(path):
    words = []
    with open(path) as fr:
        for line in fr:
            line = line.split(' ||| ')
            words.append(line[0])
    return words


def get_hypernyms(words, offset2word, word2offset):
    hypernyms = defaultdict(list)
    for w in words:
        senses = word2offset[w]
        for word_info in senses:
            h = word_info['hypernyms']
            for off in h:
                try:
                    hyper_words = offset2word[off]['words']
                except KeyError:
                    continue
                for h_w in hyper_words:
                    nb_w = len(word2offset[h_w]) * 5
                    if (h_w, nb_w) not in hypernyms[w]:
                        hypernyms[w].append((h_w, nb_w))
    return hypernyms


def write2file(hypernyms, path):
    with open(path, 'w') as fw:
        for word in hypernyms:
            hyper_words = hypernyms[word]
            hyper_words.sort(key=lambda x: x[1], reverse=True)
            hyper_words = [(w, str(n)) for w, n in hyper_words]
            fw.write("{}\t{}\n".format(
                word, '\t'.join(['\t'.join(w) for w in hyper_words])))


def main(argv=None):
    if argv is None:
        argv = sys.argv
        if len(argv) != 4:
            raise ValueError('Args Num Not Match.')
    offset2word, word2offset = read_ccd(argv[1])
    words = read_words(argv[2])
    hypernyms = get_hypernyms(words, offset2word, word2offset)
    write2file(hypernyms, argv[3])
    return 0


if __name__ == '__main__':
    sys.exit(main())
