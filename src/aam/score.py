# -*- coding: utf-8 -*-
import argparse
import sys
import numpy as np
import json
import torch
import torch.nn as nn
from torch.utils import data
from utils import to_var
from adaptive import Encoder2Decoder
import codecs


class ResDataset(data.Dataset):
    def __init__(self, file_path, word2idx, sememe2idx):
        words = []
        sememes = []
        definitions = []
        with codecs.open(file_path, 'r', 'utf-8') as fr:
            for line in fr:
                line = line.strip().split(' ||| ')
                cur_w = [word2idx[line[0]]]
                cur_s = [sememe2idx[l] for l in line[1].split(' ')]
                cur_d = [word2idx[l] for l in line[2].split(' ')]
                words.append(cur_w)
                sememes.append(cur_s)
                definitions.append(cur_d)
        sememes_padded = self.padding(sememes, 19)
        definitions_padded = self.padding(definitions, 45)
        self.words = np.array(words)
        self.sememes = np.array(sememes_padded)
        self.definitions = np.array(definitions_padded)

    def padding(self, some_list, some_len):
        new_list = []
        for l in some_list:
            l.extend([0] * (some_len - len(l)))
            if len(l) != some_len:
                print l
                print len(l)
            new_list.append(l)
        return new_list

    def __getitem__(self, index):
        word = torch.LongTensor(self.words[index])
        sememes = torch.LongTensor(self.sememes[index])
        definition = torch.LongTensor(self.definitions[index])
        return word, sememes, definition

    def __len__(self):
        return len(self.words)


def gen_score(adaptive, res_loader):
    LMcriterion = nn.CrossEntropyLoss(ignore_index=0)
    if torch.cuda.is_available():
        LMcriterion.cuda()

    adaptive.eval()
    total_scores = []
    print '--------------Start Scoring on Generated dataset---------------'
    for i, (word, sememes, definition) in enumerate(res_loader):
        word = to_var(word)
        sememes = to_var(sememes)
        definition = to_var(definition)
        targets = definition[:, 1:]

        scores, _ = adaptive(word, sememes, definition)
        scores = scores[:, :-1, :].transpose(1, 2)
        loss = LMcriterion(scores, targets)
        total_scores.append(str(np.exp(loss.data[0])))
        if (i + 1) % 10 == 0:
            print '[%s/%s]' % ((i + 1), len(res_loader))
    return total_scores


def main(args):
    with open(args.word2idx_path, 'r') as fr:
        word2idx = json.loads(fr.read())
    with open(args.sememe2idx_path, 'r') as fr:
        sememe2idx = json.loads(fr.read())
    results = ResDataset(args.gen_file_path, word2idx, sememe2idx)
    res_loader = data.DataLoader(dataset=results, batch_size=1, shuffle=False)

    if torch.cuda.is_available():
        pretrained_word_emb = torch.Tensor(
            np.load(args.pretrained_word_emb_path)).cuda()
        pretrained_sememe_emb = torch.Tensor(
            np.load(args.pretrained_sememe_emb_path)).cuda()
    else:
        pretrained_word_emb = torch.Tensor(
            np.load(args.pretrained_word_emb_path))
        pretrained_sememe_emb = torch.Tensor(
            np.load(args.pretrained_sememe_emb_path))

    # Load pretrained model or build from scratch
    adaptive = Encoder2Decoder(args.embed_size, args.hidden_size,
                               len(word2idx) + 1, pretrained_word_emb,
                               pretrained_sememe_emb)
    if torch.cuda.is_available():
        adaptive.cuda()
    adaptive.load_state_dict(torch.load(args.pretrained))
    scores = gen_score(adaptive, res_loader)
    with codecs.open(args.output_path, 'w', 'utf-8') as fw:
        fw.write('\n'.join(scores))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--gen_file_path', type=str, help='path for generated file')
    parser.add_argument(
        '--pretrained', type=str, help='path for pretrained_model')
    parser.add_argument(
        '--output_path', type=str, help='path for output directory')
    parser.add_argument(
        '--word2idx_path',
        type=str,
        default='../../data/processed/word2idx.json',
        help='path for word2idx file')
    parser.add_argument(
        '--sememe2idx_path',
        type=str,
        default='../../data/processed/sememe2idx.json',
        help='path for sememe2idx file')
    parser.add_argument(
        '--pretrained_word_emb_path',
        type=str,
        default='../../data/processed/word_matrix.npy',
        help='path for pretrained word embedding path')
    parser.add_argument(
        '--pretrained_sememe_emb_path',
        type=str,
        default='../../data/processed/sememe_matrix.npy',
        help='path for pretrained sememe embedding path')
    parser.add_argument(
        '--embed_size',
        type=int,
        default=300,
        help='dimension of word embedding vectors, also dimension of v_g')
    parser.add_argument(
        '--hidden_size',
        type=int,
        default=512,
        help='dimension of lstm hidden states')
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument(
        '--num_workers', type=int, default=4, help='number of cpu workers')
    args = parser.parse_args()
    sys.exit(main(args))
