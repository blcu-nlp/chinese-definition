import torch
import numpy as np
import torch.nn as nn
from data_loader import get_loader
from torch.autograd import Variable


# Variable wrapper
def to_var(x, volatile=False):
    '''
    Wrapper torch tensor into Variable
    '''
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


# Evaluation function
def defseq_eval(model, args, epoch):
    '''
    model: trained model to be evaluated
    args: pre-set parameters
    epoch: epoch #, for disp purpose
    '''
    model.eval()
    eval_data_loader = get_loader(
        args.valid_path,
        args.eval_size,
        shuffle=False,
        num_workers=args.num_workers,
        mode='valid')

    LMcriterion = nn.CrossEntropyLoss(ignore_index=0)
    if torch.cuda.is_available():
        LMcriterion.cuda()
    eval_loss = []
    print '--------------Start evaluation on Validation dataset---------------'
    for i, (word, sememes, definition) in enumerate(eval_data_loader):
        word = to_var(word)
        sememes = to_var(sememes)
        definition = to_var(definition)
        targets = definition[:, 1:]

        scores, _ = model(word, sememes, definition)
        scores = scores[:, :-1, :].transpose(1, 2)
        loss = LMcriterion(scores, targets)
        eval_loss.append(loss.data[0])

    valid_ppl = np.exp(np.mean(eval_loss))
    return valid_ppl
