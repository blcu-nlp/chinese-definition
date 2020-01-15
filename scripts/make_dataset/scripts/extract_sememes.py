# -*- coding: utf-8 -*-

import re
import sys
from collections import defaultdict


def read_hownet(hownet_path):
    word_sememes = defaultdict(list)
    with open(hownet_path) as fr:
        for line in fr:
            line = line.strip()
            if 'W_C=' in line:
                word = line.split('=')[1]
            if 'DEF=' in line:
                sememes = re.findall(r"\w+\|\w+", line)
                print(word, ' ', sememes)
                if sememes not in word_sememes[word]:
                    word_sememes[word].append(sememes)
    return word_sememes


def write2file(word_sememes, save_path):
    with open(save_path, 'w') as fw:
        for word, sememes in word_sememes.items():
            for s in sememes:
                fw.write("{} ||| {}\n".format(word, ' '.join(s)))


def main(argv=None):
    if argv is None:
        argv = sys.argv
        if len(argv) != 3:
            print('Args Error!')
            return 1
    hownet_path = argv[1]
    save_path = argv[2]
    word_sememes = read_hownet(hownet_path)
    write2file(word_sememes, save_path)
    return 0


if __name__ == '__main__':
    sys.exit(main())
