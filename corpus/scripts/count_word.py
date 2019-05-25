# -*- coding: utf-8 -*-

import sys
import os
import xlrd


def ccd_words(csv_path):
    word_set = set()
    ccd_file = 'CCD_Data_{}.xls'
    pos = ['Adj', 'Adv', 'Noun', 'Verb']
    for p in pos:
        ccd_file_path = os.path.join(csv_path, ccd_file.format(p))
        data = xlrd.open_workbook(ccd_file_path)
        table = data.sheets()[0]
        nrows = table.nrows
        for i in range(1, nrows):
            line = table.row_values(i)
            print(p, ' ', i)
            print(line)
            words = line[4].split(' ')
            for w in words:
                # if w.startswith('"'):
                    # print(content)
                word_set.add(w)
    return word_set


def write2file(word_set, out_file):
    word_list = sorted(word_set)
    with open(out_file, 'w') as fw:
        fw.write('\n'.join(word_list))


def main(argv=None):
    if argv is None:
        argv = sys.argv
        if len(argv) != 3:
            print('Arg Error!')
            return 1
    csv_path = argv[1]
    out_file = argv[2]
    word_set = ccd_words(csv_path)
    write2file(word_set, out_file)
    return 0


if __name__ == '__main__':
    sys.exit(main())
