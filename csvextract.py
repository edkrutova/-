#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
csvextract.py: Extract lines from csv file(s) utility
"""

import csv
import os


def compose_filepath(dirname, filename):
    if dirname:
        if not os.path.isdir(dirname):
            os.mkdir(dirname)
        return os.path.join(dirname, filename)
    return filename

def limited_increment(n, limit, value=1):
    n += value;
    if n > limit:
        n = 0
    return n

first_file = True

def process_file(args, filepath):
    global first_file
    i = 0
    print('Process file', filepath)
    with open(filepath, newline='') as f:
        reader = csv.DictReader(f)
        if args.column is None:
            open_mode = 'w' if first_file else 'a'
            all_texts = {}
            if args.all:
                fdc = open(compose_filepath(args.output, 'all.csv'), open_mode)
            fds = []
            fds.append(open(compose_filepath(args.output, 'test.csv'), open_mode))
            fds.append(open(compose_filepath(args.output, 'train.csv'), open_mode))
            fds.append(open(compose_filepath(args.output, 'validation.csv'), open_mode))
            if first_file:
                for fd in fds:
                    print(f'"log","label"', file=fd)
            first_file = False
            nx_0 = 0
            nx_1 = 0
            label_0_line_no = 0
        for row in reader:
            if args.column:
                print(row[args.column])
            else:
                label = '0' if row['Label'] == 'Normal' else '1'
                text = row['LogText']
                # print(f'"{text}","{label}"')
                if not text in all_texts: # skip duplicate texts
                    if '"' in text:
                        # print('AAA:', text)
                        text = text.replace('"', "'")
                        # print('BBB:', text)
                    if args.all:
                        print(f'"{text}","{label}"', file=fdc)
                    if label == '0':
                        if label_0_line_no % args.each == 0:
                            print(f'"{text}","{label}"', file=fds[nx_0])
                            nx_0 = limited_increment(nx_0, 2)
                        label_0_line_no += 1
                    elif label == '1':
                        print(f'"{text}","{label}"', file=fds[nx_1])
                        nx_1 = limited_increment(nx_1, 2)
                    all_texts[text] = 1
            i += 1
            if args.nlines > 0 and i > args.nlines:
                break
        if args.column is None:
            for fd in fds:
                fd.close()
            if args.all:
                fdc.close()

def print_labels(filepath):
    print(filepath, 'labels:')
    with open(filepath) as f:
        for line in f:
            print(line.rstrip())
            return


if __name__ == '__main__':
    import sys

    from argparse import ArgumentParser

    parser = ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument('filelist', nargs='*', metavar='NAME', type=str, help='file to process')
    parser.add_argument('-A', '--all', action='store_const', const=True, default=False, help='save file all.cvs also')
    parser.add_argument('-L', '--labels', action='store_const', const=True, default=False, help='print column labels and exit')
    parser.add_argument('-d', '--debug', metavar='LEVEL', type=int, default=1, help='debug level, default is 1')
    parser.add_argument('-c', '--column', metavar='NAME', type=str, help='print single column')
    parser.add_argument('-e', '--each', metavar='N', type=int, nargs='?', default=1, help='use each N-th zero-labeled line, default is all')
    parser.add_argument('-n', '--nlines', metavar='N', type=int, nargs='?', default=0, help='process first N lines, default is all')
    parser.add_argument('-o', '--output', metavar='DIR', default='', help='save cvs files to DIR')

    args = parser.parse_args()

    if not args.filelist:
        print('ERROR: Needs at least one filename as argument')
        sys.exit(0)

    if args.labels:
        for filepath in args.filelist:
            print_labels(filepath)
        sys.exit(0)
    try:
        for filepath in args.filelist:
            process_file(args, filepath)
    except KeyboardInterrupt:
        print('\x08\x08\x08\x08*** Interrupted by Ctrl-C ***')
        sys.exit(1)
