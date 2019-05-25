from __future__ import print_function
# import math
import json
import argparse
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm
import numpy as np
import os
import sys
from utils import defseq_eval, to_var
from data_loader import get_loader
from adaptive import Encoder2Decoder

# from torch.nn.utils.rnn import pack_padded_sequence

# from build_vocab import Vocabulary
# from torch.autograd import Variable
# from torchvision import transforms


def main(args):

    # To reproduce training results
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    # Load word2idx
    with open(args.word2idx_path, 'r') as fr:
        word2idx = json.loads(fr.read())

    # Build training data loader
    data_loader = get_loader(
        args.train_path,
        args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        mode='train')

    # Load pretrained embeddings
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

    if args.pretrained:
        adaptive.load_state_dict(torch.load(args.pretrained))
        # Get starting epoch #,
        # note that model is named as
        # '...your path to model/algoname-epoch#.pkl'
        # A little messy here.
        start_epoch = int(
            args.pretrained.split('/')[-1].split('-')[1].split('.')[0]) + 1
    else:
        start_epoch = 1

    # Will decay later
    # learning_rate = args.learning_rate

    # Language Modeling Loss
    LMcriterion = nn.CrossEntropyLoss(ignore_index=0)

    # Change to GPU mode if available
    if torch.cuda.is_available():
        adaptive.cuda()
        LMcriterion.cuda()

    # Train the Models
    total_step = len(data_loader)

    ppl_scores = []
    best_ppl = 0.0
    best_epoch = 0

    # Start Learning Rate Decay
    # if epoch > args.lr_decay:

    #     frac = float(epoch -
    #                     args.lr_decay) / args.learning_rate_decay_every
    #     decay_factor = math.pow(0.5, frac)

    #     # Decay the learning rate
    #     learning_rate = args.learning_rate * decay_factor

    # print('Learning Rate for Epoch %d: %.6f' % (epoch, learning_rate))

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, adaptive.parameters()),
        lr=args.learning_rate,
        betas=(args.alpha, args.beta))

    # Start Training
    for epoch in range(start_epoch, args.num_epochs + 1):

        epoch_loss = []

        # Language Modeling Training
        print('------------------Training for Epoch %d----------------' %
              (epoch))
        for i, (word, sememes, definition) in enumerate(data_loader):
            # Set mini-batch dataset
            word = to_var(word)
            sememes = to_var(sememes)
            definition = to_var(definition)
            targets = definition[:, 1:]

            # Forward, Backward and Optimize
            adaptive.train()
            adaptive.zero_grad()

            scores, _ = adaptive(word, sememes, definition)
            scores = scores[:, :-1, :].transpose(1, 2)

            # Compute loss and backprop
            loss = LMcriterion(scores, targets)
            epoch_loss.append(loss.data[0])
            loss.backward()

            # Gradient clipping for gradient exploding problem in LSTM
            # for p in adaptive.decoder.LSTM.parameters():
            #     p.data.clamp_(-args.clip, args.clip)

            clip_grad_norm(
                filter(lambda p: p.requires_grad, adaptive.parameters()),
                args.clip)
            # print(args.clip)
            optimizer.step()

            # Print log info
            if i % args.log_step == 0:
                print(
                    'Epoch [%d/%d], Step [%d/%d], CrossEntropy Loss: %.4f, Perplexity: %5.4f'
                    % (epoch, args.num_epochs, i, total_step, loss.data[0],
                       np.exp(loss.data[0])))
        train_loss = np.mean(epoch_loss)
        train_ppl = np.exp(train_loss)
        # Save the Adaptive Attention model after each epoch
        torch.save(adaptive.state_dict(),
                   os.path.join(args.model_path, 'adaptive-%d.pkl' % (epoch)))

        # Evaluation on validation set
        valid_ppl = defseq_eval(adaptive, args, epoch)
        ppl_scores.append(valid_ppl)

        print(
            'Epoch [%d/%d], Train Loss: %.4f, Train PPL: %5.4f, Valid PPL: %5.4f'
            % (epoch, args.num_epochs, train_loss, train_ppl, valid_ppl))

        if valid_ppl < best_ppl or best_ppl == 0.0:
            best_ppl = valid_ppl
            best_epoch = epoch

        if len(ppl_scores) > 5:
            last_6 = ppl_scores[-6:]
            last_6_min = min(last_6)

            # Test if there is improvement, if not do early stopping
            if last_6_min != best_ppl:

                print(
                    'No improvement with ppl in the last 6 epochs...Early stopping triggered.'
                )
                print('Model of best epoch #: %d with ppl score %.2f' %
                      (best_epoch, best_ppl))
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-f', default='self', help='To make it runnable in jupyter')
    parser.add_argument(
        '--model_path',
        type=str,
        default='../models/adaptive/',
        help='path for saving trained models')
    parser.add_argument(
        '--word2idx_path',
        type=str,
        default='../../data/processed/word2idx.json',
        help='path for word2idx file')
    parser.add_argument(
        '--train_path',
        type=str,
        default='../../data/processed/train.npz',
        help='path for train dataset')
    parser.add_argument(
        '--valid_path',
        type=str,
        default='../../data/processed/valid.npz',
        help='path for valid dataset')
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
        '--log_step',
        type=int,
        default=10,
        help='step size for printing log info')
    parser.add_argument(
        '--seed',
        type=int,
        default=1024,
        help='random seed for model reproduction')

    # --------Hyper Parameter Setup------------------

    # Optimizer Adam parameter
    parser.add_argument(
        '--alpha', type=float, default=0.8, help='alpha in Adam')
    parser.add_argument(
        '--beta', type=float, default=0.999, help='beta in Adam')
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=4e-4,
        help='learning rate for the whole model')

    # LSTM hyper parameters
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

    # Training details
    parser.add_argument(
        '--pretrained',
        type=str,
        default='',
        help='start from checkpoint or scratch')
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument(
        '--batch_size', type=int,
        default=50)  # on cluster setup, 60 each x 4 for Huckle server

    # For eval_size > 30, it will cause cuda OOM error on Huckleberry
    parser.add_argument(
        '--eval_size', type=int, default=30)  # on cluster setup, 30 each x 4
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--clip', type=float, default=0.1)
    parser.add_argument(
        '--lr_decay',
        type=int,
        default=20,
        help='epoch at which to start lr decay')
    parser.add_argument(
        '--learning_rate_decay_every',
        type=int,
        default=50,
        help='decay learning rate at every this number')

    args = parser.parse_args()

    print('-----------Model and Training Details------------')
    print(args)

    # Start training
    sys.exit(main(args))
