#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import sys


def read_file(path):
    content = []
    with open(path) as fr:
        for line in fr:
            content.append(line.strip())
    return content


def arrange(src, tgt):
    lines = []
    set_s = set()
    assert len(src) == len(tgt), 'src and tgt length not equal'
    for s, t in zip(src, tgt):
        if s in set_s:
            continue
        else:
            set_s.add(s)
        s = s.split(' ')
        w = s[0]
        s = ' '.join(s[1:])
        line = ' ||| '.join([w, s, t])
        lines.append(line)
    return lines


def write2file(lines, path):
    with open(path, 'w') as fw:
        fw.write('\n'.join(lines))
    return 1


def concator(argv=None):
    if argv is None:
        argv = sys.argv[1:]
        if len(argv) != 3:
            raise ValueError("Usage: <src> <tgt> <out>")
    src = read_file(argv[0])
    tgt = read_file(argv[1])
    lines = arrange(src, tgt)
    if write2file(lines, argv[2]):
        print('Wrote to -->', argv[2])
    return 0


if __name__ == '__main__':
    sys.exit(concator())
