# -*- coding: utf-8 -*-
import numpy as np


class BeamSearch(object):
    """return beam_size samples and their NLL scores,
    each sample is a sequence of labels,
    all samples starts with an `bos` label and end with `eos` or
    truncated to length of `maxsample`.
    usage example:
    beam = BeamSearch(400, 0, 1, 2, 10)
    beam.reset()
    probs = model(inp)
    while beam.beam(probs):
        model(beam.live_samples)
    print(''.join(beam.dead_samples))
    """

    def __init__(self, max_len, unk, bos, eos, beam_size=1):
        self.max_len = max_len
        self.unk = unk
        self.bos = bos
        self.eos = eos
        self.beam_size = beam_size

    def reset(self):
        self.dead_k = 0
        self.dead_samples = []
        self.dead_scores = []
        self.live_k = 1
        self.live_samples = [[self.bos]]
        self.live_scores = [0]

    def beam(self, probs):
        """
        probs: tensor have shape (seq_len, batch_size, nb_class);
        output: a boolean value, continue or not;
        """
        # total score for every sample is sum of -log of word prb
        cand_scores = np.array(self.live_scores)[:, None] - np.array(probs)
        cand_scores[:, self.unk] = 1e20
        cand_flat = cand_scores.flatten()

        # find the best (lowest) scores we have
        # from all possible samples and new words
        ranks_flat = cand_flat.argsort()[:(self.beam_size - self.dead_k)]
        self.live_scores = cand_flat[ranks_flat]

        # append the new words to their appropriate live sample
        voc_size = probs.shape[1]
        self.live_samples = [
            self.live_samples[r // voc_size] + [r % voc_size]
            for r in ranks_flat
        ]

        # live samples that should be dead are...
        zombie = [
            s[-1] == self.eos or len(s) >= self.max_len
            for s in self.live_samples
        ]

        # add zombies to the dead
        self.dead_samples += [
            s for s, z in zip(self.live_samples, zombie) if z
        ]  # remove first label == empty
        self.dead_scores += [s for s, z in zip(self.live_scores, zombie) if z]
        self.dead_k = len(self.dead_samples)
        # remove zombies from the living
        self.live_samples = [
            s for s, z in zip(self.live_samples, zombie) if not z
        ]
        self.live_scores = [
            s for s, z in zip(self.live_scores, zombie) if not z
        ]
        self.live_k = len(self.live_samples)
        still = bool(self.live_k and self.dead_k < self.beam_size)
        self.output = self.dead_samples + self.live_samples
        self.output_scores = self.dead_scores + self.live_scores
        return still
