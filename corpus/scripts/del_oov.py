# -*- coding: utf-8 -*-
import sys


def read_word_list(path):
    word_list = set()
    with open(path) as fr:
        for line in fr:
            word_list.add(line.strip())
    return word_list


def read_dataset(word_list, path):
    dataset = []
    with open(path) as fr:
        for line in fr:
            line = line.strip()
            if line.split(' ||| ')[0] in word_list:
                dataset.append(line)
    return dataset


def write2file(dataset, path):
    file_path = path + '.new'
    with open(file_path, 'w') as fw:
        fw.write('\n'.join(dataset))


def main(argv=None):
    if argv is None:
        argv = sys.argv
        if len(argv) != 3:
            raise ValueError("Args Num Not Match.")
    del_word_list = read_word_list(argv[1])
    dataset = read_dataset(del_word_list, argv[2])
    write2file(dataset, argv[2])
    return 0


if __name__ == '__main__':
    sys.exit(main())
