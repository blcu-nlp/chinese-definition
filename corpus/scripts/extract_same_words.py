# -*- coding: utf-8 -*-
import sys


def read_word_list(path):
    word_set = set()
    with open(path) as fr:
        for line in fr:
            word_set.add(line.strip())
    return word_set


def main(argv=None):
    if argv is None:
        argv = sys.argv
        if len(argv) != 4:
            raise ValueError("Args Num Not Match.")
    word_set_a = read_word_list(argv[1])
    word_set_b = read_word_list(argv[2])
    a_and_b = word_set_a & word_set_b
    with open(argv[3], 'w') as fw:
        fw.write('\n'.join(sorted(a_and_b)))
    return 0


if __name__ == '__main__':
    sys.exit(main())