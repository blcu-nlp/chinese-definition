# -*- coding: utf-8 -*-

import re
import sys
from collections import defaultdict


def read_giga_list(path):
    giga_freq = {}
    with open(path) as fr:
        for line in fr:
            line = line.strip().split(' ')
            giga_freq[line[0]] = line[1]
    return giga_freq


def read_same_words(path):
    same_words = []
    with open(path) as fr:
        for line in fr:
            line = line.strip()
            if len(line) == 1 or re.search(r"^[0-9, a-z, A-Z]", line):
                continue
            same_words.append(line.strip())
    return same_words


def read_hownet(path):
    hownet = defaultdict(list)
    with open(path) as fr:
        for line in fr:
            line = line.strip().split(' ||| ')
            if len(line) != 2:
                continue
            word = line[0]
            sememes = line[1].split(' ')
            hownet[word].append(sememes)
    return hownet


def sort_words(giga_freq, words, hownet):
    filter_sememes = [
        'cardinal|基数', 'ordinal|序数', 'FuncWord|功能词', 'ProperName|专',
        'biology|生物学', 'math|数学', 'chemical|化学物', 'physics|物理',
        'physiology|生理学', 'politics|政', 'language|语言', 'stone|土石',
        'chemistry|化学'
    ]
    filter_start_with = [
        'FlowerGrass|花草', 'disease|疾病', 'InsectWorm|虫', 'beast|走兽', 'tree|树',
        'fish|鱼', 'bird|禽', 'crop|庄稼', 'medicine|药物', 'AlgaeFungi|低植',
        'bacteria|微生物', 'celestial|天体', 'metal|金属', 'Unit|单位', 'fruit|水果',
        'money|货币'
    ]
    filter_as_one = [['human|人', 'RelatingToCountry|与特定国家相关']]
    words_match = []
    words_not_match = []
    for w in words:
        ignore_word = False
        sememes = hownet[w]
        for sememe in sememes:
            for s in filter_as_one:
                if s == sememe:
                    ignore_word = True
            for s in filter_sememes:
                if s in sememe:
                    ignore_word = True
            for s in filter_start_with:
                if s == sememe[0]:
                    ignore_word = True
        if ignore_word:
            continue
        if w in giga_freq:
            freq = int(giga_freq[w])
            words_match.append((w, freq))
        else:
            words_not_match.append(w)

    words_match.sort(key=lambda x: x[1], reverse=True)
    print('matched: ', len(words_match))
    print('not matched: ', len(words_not_match))
    sorted_words = [w[0] for w in words_match]

    # with open('matched.txt', 'w') as fw_m:
    #     fw_m.write('\n'.join(sorted_words))
    # with open('not_match.txt', 'w') as fw_nm:
    #     fw_nm.write('\n'.join(words_not_match))

    sorted_words.extend(words_not_match)
    return sorted_words


def write2file(path, sorted_words):
    with open(path, 'w') as fw:
        fw.write('\n'.join(sorted_words))


def main(argv=None):
    if argv is None:
        argv = sys.argv
        if len(argv) != 5:
            raise ValueError("Arg nums not match.")
    giga_freq = read_giga_list(argv[1])
    same_words = read_same_words(argv[2])
    hownet = read_hownet(argv[3])
    write_path = argv[4]
    sorted_words = sort_words(giga_freq, same_words, hownet)
    write2file(write_path, sorted_words)
    return 0


if __name__ == '__main__':
    sys.exit(main())
