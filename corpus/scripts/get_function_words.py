# -*- coding: utf-8 -*-
import sys
from collections import defaultdict


def read_hownet(hownet_file):
    hownet = defaultdict(list)
    with open(hownet_file) as fr:
        for line in fr:
            line = line.strip().split('\t')
            if len(line) == 2:
                sense = line[1].split(' ')
                hownet[line[0]].append(sense)
    return hownet


def filter_funcwords(hownet):
    funcwords = set()
    for word in hownet:
        for d in hownet[word]:
            if 'FuncWord|功能词' in d:
                if word not in funcwords:
                    funcwords.add(word)
    return funcwords


def write2file(funcwords, path):
    with open(path, 'w') as fw:
        fw.write('\n'.join(funcwords))


def main(argv=None):
    if argv is None:
        argv = sys.argv
        if len(argv) != 3:
            raise ValueError("Args Num Not Match.")
    hownet = read_hownet(argv[1])
    funcwords = filter_funcwords(hownet)
    write2file(funcwords, argv[2])


if __name__ == '__main__':
    sys.exit(main())
