# -*- coding: utf-8 -*-
import numpy as np
import json
import sys
import os
import argparse


def read_data(path):
    content = []
    with open(path) as fr:
        for line in fr:
            line = line.strip()
            if line == '':
                continue
            content.append(line.split(' ||| '))
    return content


def make_idx(inp_train_path, inp_valid_path, inp_test_path):
    dataset = []
    dataset.extend(read_data(inp_train_path))
    dataset.extend(read_data(inp_valid_path))
    dataset.extend(read_data(inp_test_path))
    word_freq = {}
    for _, _, definition in dataset:
        for w in definition.split(' '):
            if w not in word_freq:
                word_freq[w] = 0
            word_freq[w] += 1

    emb_vocab_path = os.path.join(args.inp_data_dir, args.emb_vocab)
    emb_vocab = set()
    with open(emb_vocab_path) as fr:
        for line in fr:
            if line.strip() == '':
                continue
            emb_vocab.add(line.strip())

    word2idx = {}
    max_def_len = 0
    word2idx[args.bos] = 1
    word2idx[args.eos] = 2
    word2idx[args.unk] = 3
    words = [d[0] for d in dataset]
    definitions = [d[2] for d in dataset]
    for w in words:
        if w not in word2idx:
            word2idx[w] = len(word2idx) + 1

    for d in definitions:
        max_def_len = max(max_def_len, len(d.split(' ')))
        for w in d.split(' '):
            if w not in word2idx:
                if w == '':
                    continue
                if word_freq[w] < args.unk_thres or w not in emb_vocab:
                    word2idx[w] = word2idx[args.unk]
                else:
                    word2idx[w] = len(word2idx) + 1

    idx2word = {v: k for k, v in word2idx.iteritems()}
    idx2word[3] = args.unk
    return word2idx, idx2word, max_def_len


def make_sememe_idx(inp_train_path, inp_valid_path, inp_test_path):
    dataset = []
    dataset.extend(read_data(inp_train_path))
    dataset.extend(read_data(inp_valid_path))
    dataset.extend(read_data(inp_test_path))
    sememe2idx = {}
    max_sem_len = 0
    for _, sememes, _ in dataset:
        sememe_list = sememes.split(' ')
        max_sem_len = max(max_sem_len, len(sememe_list))
        for s in sememe_list:
            if s not in sememe2idx:
                sememe2idx[s] = len(sememe2idx) + 1
    idx2sememe = {v: k for k, v in sememe2idx.iteritems()}
    return sememe2idx, idx2sememe, max_sem_len


def padding(some_list, some_len):
    new_list = []
    for l in some_list:
        l.extend([0] * (some_len - len(l)))
        if len(l) != some_len:
            print l
            print len(l)
        new_list.append(l)
    return new_list


def make_dataset(inp_path, word2idx, sememe2idx, max_def_len, max_sem_len):
    data_content = read_data(inp_path)
    word_list = []
    sememe_list = []
    definition_list = []
    for word, sememes, definition in data_content:
        word_list.append([word2idx[word]])
        sememes = [sememe2idx[s] for s in sememes.split(' ')]
        sememe_list.append(sememes)
        definition = [word2idx[d] for d in definition.split(' ') if d != '']
        definition.insert(0, word2idx[args.bos])
        definition.append(word2idx[args.eos])
        definition_list.append(definition)
    sememe_list_padding = padding(sememe_list, max_sem_len)
    definition_list_padding = padding(definition_list, max_def_len + 2)
    word_array = np.array(word_list)
    sememe_array = np.array(sememe_list_padding)
    definition_array = np.array(definition_list_padding)
    print 'Word Array: ', word_array.shape
    print 'Sememe Array: ', sememe_array.shape
    print 'Definition Array: ', definition_array.shape
    data = [word_array, sememe_array, definition_array]
    return data


def make_test_dataset(inp_path, word2idx, sememe2idx, max_sem_len):
    data_content = read_data(inp_path)
    word_list = []
    sememe_list = []
    word_sememe_set = set()
    for word, sememes, definition in data_content:
        if (word, sememes) in word_sememe_set:
            continue
        else:
            word_sememe_set.add((word, sememes))
        word_list.append([word2idx[word]])
        sememes = [sememe2idx[s] for s in sememes.split(' ')]
        sememe_list.append(sememes)
    sememe_list_padding = padding(sememe_list, max_sem_len)
    word_array = np.array(word_list)
    sememe_array = np.array(sememe_list_padding)
    print 'Word Array: ', word_array.shape
    print 'Sememe Array: ', sememe_array.shape
    data = [word_array, sememe_array]
    return data


def main(args):
    if not os.path.exists(args.save_data_dir):
        os.makedirs(args.save_data_dir)
    inp_train_path = os.path.join(args.inp_data_dir, args.inp_train)
    inp_valid_path = os.path.join(args.inp_data_dir, args.inp_valid)
    inp_test_path = os.path.join(args.inp_data_dir, args.inp_test)
    save_train_path = os.path.join(args.save_data_dir, args.save_train)
    save_valid_path = os.path.join(args.save_data_dir, args.save_valid)
    save_test_path = os.path.join(args.save_data_dir, args.save_test)
    save_word2idx_path = os.path.join(args.save_data_dir, args.save_word2idx)
    save_idx2word_path = os.path.join(args.save_data_dir, args.save_idx2word)
    save_sememe2idx_path = os.path.join(args.save_data_dir,
                                        args.save_sememe2idx)
    save_idx2sememe_path = os.path.join(args.save_data_dir,
                                        args.save_idx2sememe)
    print 'Making Index...'
    word2idx, idx2word, max_def_len = make_idx(inp_train_path, inp_valid_path,
                                               inp_test_path)
    with open(save_word2idx_path, 'w') as fw:
        fw.write(json.dumps(word2idx))
    with open(save_idx2word_path, 'w') as fw:
        fw.write(json.dumps(idx2word))
    sememe2idx, idx2sememe, max_sem_len = make_sememe_idx(
        inp_train_path, inp_valid_path, inp_test_path)
    with open(save_sememe2idx_path, 'w') as fw:
        fw.write(json.dumps(sememe2idx))
    with open(save_idx2sememe_path, 'w') as fw:
        fw.write(json.dumps(idx2sememe))
    print 'Making Train Dataset...'
    train_data = make_dataset(inp_train_path, word2idx, sememe2idx,
                              max_def_len, max_sem_len)
    print 'Making Valid Dataset...'
    valid_data = make_dataset(inp_valid_path, word2idx, sememe2idx,
                              max_def_len, max_sem_len)
    print 'Making Test Dataset...'
    test_data = make_test_dataset(inp_test_path, word2idx, sememe2idx,
                                  max_sem_len)
    np.savez(
        save_train_path,
        words=train_data[0],
        sememes=train_data[1],
        definitions=train_data[2])
    np.savez(
        save_valid_path,
        words=valid_data[0],
        sememes=valid_data[1],
        definitions=valid_data[2])
    np.savez(save_test_path, words=test_data[0], sememes=test_data[1])
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bos', type=str, default='<s>')
    parser.add_argument('--eos', type=str, default='</s>')
    parser.add_argument('--unk', type=str, default='UNK')
    parser.add_argument('--unk_thres', type=int, default=5)
    parser.add_argument('--inp_data_dir', type=str, default='../../data')
    parser.add_argument('--emb_vocab', type=str, default='emb_vocab.txt')
    parser.add_argument('--inp_train', type=str, default='train.txt')
    parser.add_argument('--inp_valid', type=str, default='valid.txt')
    parser.add_argument('--inp_test', type=str, default='test.txt')
    parser.add_argument(
        '--save_data_dir', type=str, default='../data/processed')
    parser.add_argument('--save_train', type=str, default='train.npz')
    parser.add_argument('--save_valid', type=str, default='valid.npz')
    parser.add_argument('--save_test', type=str, default='test.npz')
    parser.add_argument('--save_word2idx', type=str, default='word2idx.json')
    parser.add_argument('--save_idx2word', type=str, default='idx2word.json')
    parser.add_argument(
        '--save_sememe2idx', type=str, default='sememe2idx.json')
    parser.add_argument(
        '--save_idx2sememe', type=str, default='idx2sememe.json')
    args = parser.parse_args()
    sys.exit(main(args))
