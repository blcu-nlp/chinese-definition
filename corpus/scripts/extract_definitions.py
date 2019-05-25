# -*- coding: utf-8 -*-

import os
import re
import sys
from collections import defaultdict

import xlrd


def get_ccd_def(ccd_file_path):
    word_definitions = defaultdict(list)
    ccd_file_format = 'CCD_Data_{}.xls'
    pos = ['Adj', 'Adv', 'Noun', 'Verb']
    for p in pos:
        ccd_file = ccd_file_format.format(p)
        print(ccd_file)
        xls_file = os.path.join(ccd_file_path, ccd_file)
        data = xlrd.open_workbook(xls_file)
        table = data.sheets()[0]
        nrows = table.nrows
        for i in range(1, nrows):
            line = table.row_values(i)
            real_p = line[1]
            category = line[2]
            words = line[4].split(' ')
            definition = line[6]
            definition = re.sub(r"（.*?）", "", definition)
            definition = re.sub(r"（.*?", "", definition)
            definition = re.sub(r".*?）", "", definition)
            definition = re.sub(r"“.*?”", "", definition)
            if '\r\n' in definition:
                definition = definition.split('\r\n')[0]
            definition = re.sub(r"\r", "", definition)
            for w in words:
                if w not in definition\
                        and w != 'X'\
                        and definition != 'X'\
                        and definition != "":
                    word_def = {
                        'pos': real_p,
                        'category': category,
                        'definition': definition
                    }
                    if word_def not in word_definitions[w]:
                        word_definitions[w].append(word_def)
    return word_definitions


def extract_defs(words_path, word_definitions):
    extracted_word_defs = defaultdict(list)
    word_list = open(words_path).readlines()
    word_list = [w.strip() for w in word_list if w.strip() != '']
    for w in word_list:
        extracted_word_defs[w] = word_definitions[w]
    return extracted_word_defs


def write2file(extracted_word_defs, definition_file):
    with open(definition_file, 'w') as fw:
        for key, entries in extracted_word_defs.items():
            for e in entries:
                fw.write("{} ||| {} ||| {} ||| {}\n".format(
                    key, e['pos'], e['category'], e['definition']))


def main(argv=None):
    if argv is None:
        argv = sys.argv
        if len(argv) != 4:
            print('Args Error!')
            return 1
    ccd_file_path = argv[1]
    words_path = argv[2]
    definition_file = argv[3]
    word_definitions = get_ccd_def(ccd_file_path)
    extracted_word_defs = extract_defs(words_path, word_definitions)
    write2file(extracted_word_defs, definition_file)
    return 0


if __name__ == '__main__':
    sys.exit(main())
