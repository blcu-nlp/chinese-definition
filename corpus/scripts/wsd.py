# -*- coding: utf-8 -*-
import sys
from collections import defaultdict
from progressbar import progressbar
import jieba


def read_ccd(ccd_file):
    ccd = defaultdict(list)
    with open(ccd_file) as fr:
        for line in fr:
            line = line.strip().split(' ||| ')
            info = {'pos': line[1], 'category': line[2], 'definition': line[3]}
            ccd[line[0]].append(info)
    return ccd


def read_hownet(hownet_file):
    hownet = defaultdict(list)
    with open(hownet_file) as fr:
        for line in fr:
            line = line.strip().split(' ||| ')
            if len(line) == 2:
                sense = line[1].split(' ')
                hownet[line[0]].append(sense)
    return hownet


def sim_score(definition, sememes):
    sent_sememes = []
    score = 0
    for word in jieba.lcut(definition):
        if word in hownet:
            senses = hownet[word]
            for s in senses:
                sent_sememes.extend(s)
    for s in sent_sememes:
        if s in sememes:
            score += 1
    return score


def wsd_on_hownet(ccd):
    ccd_with_sememes = defaultdict(list)
    for word in progressbar(ccd):
        senses = hownet[word]
        info = ccd[word]
        if len(info) == len(senses) == 1:
            if sim_score(info[0]['definition'], senses[0]) > 0:
                sem_info = info[0]
                sem_info['sememes'] = senses[0]
                ccd_with_sememes[word].append(sem_info)
        else:
            score_matrix = []
            used_info_index = []
            max_nb_senses = len(info)
            for idx_s, s in enumerate(senses):
                for idx_i, i in enumerate(info):
                    score = sim_score(i['definition'], s)
                    score_matrix.append((idx_s, idx_i, score))
            score_matrix.sort(key=lambda x: x[2], reverse=True)

            for score in score_matrix:
                if len(ccd_with_sememes[word]) >= max_nb_senses:
                    break
                if score[1] in used_info_index:
                    continue
                if score[2] != 0:
                    sem_info = info[score[1]]
                    sem_info['sememes'] = senses[score[0]]
                    if sem_info not in ccd_with_sememes[word]:
                        ccd_with_sememes[word].append(sem_info)
                        used_info_index.append(score[1])

    ccd_with_sememes_split_defs = defaultdict(list)
    for word in ccd_with_sememes:
        info = ccd_with_sememes[word]
        for i in info:
            sem_info = i
            if '；' in i['definition']:
                for d in i['definition'].split('；'):
                    sem_info['definition'] = d
                    if sem_info not in ccd_with_sememes_split_defs[word]:
                        ccd_with_sememes_split_defs[word].append(sem_info)
            else:
                if sem_info not in ccd_with_sememes_split_defs[word]:
                    ccd_with_sememes_split_defs[word].append(sem_info)
    return ccd_with_sememes_split_defs


def write2file(ccd_with_sememes, save_file):
    with open(save_file, 'w') as fw:
        for word in ccd_with_sememes:
            info = ccd_with_sememes[word]
            for i in info:
                fw.write("{} ||| {} ||| {} ||| {} ||| {}\n".format(
                    word, i['pos'], i['category'], ' '.join(i['sememes']),
                    i['definition']))


def main(argv=None):
    if argv is None:
        argv = sys.argv
        if len(argv) != 4:
            raise ValueError("Wrong Arguments.")
    ccd = read_ccd(argv[1])
    global hownet
    hownet = read_hownet(argv[2])
    save_file = argv[3]
    print('Loading CCD...')
    print('Loading HowNet...')
    print('WSD:')
    ccd_with_sememes = wsd_on_hownet(ccd)
    write2file(ccd_with_sememes, save_file)
    return 0


if __name__ == '__main__':
    sys.exit(main())
