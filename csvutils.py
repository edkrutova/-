# -*- coding: utf-8 -*-
"""
csvutils.py: csv utilities
"""

import re
import sys
import csv


re_pattern_core = re.compile(r'core.[0-9]*')
re_pattern_address = re.compile(r'address=[0-9a-fA-F\.\,\)xX]*')
re_pattern_unit = re.compile(r'unit=[0-9a-fA-F\.\,\)xX]*')
re_pattern_bit = re.compile(r'bit=[0-9a-fA-F\.\,\)xX]*')
re_pattern_number = re.compile(r'[0-9a-fA-F\.\,\)xX]+$')

def clean_digits(s):
    rv = ''
    for w in s.split():
        if w.isdigit():
            rv += f' N'
        elif re_pattern_number.match(w):
            rv += f' X'
        else:
            rv += f' {w}'
    # print(a)
    return rv.lstrip()

def clean_line(s):
    start_tokens = ['RAS', 'DISCOVERY', 'HARDWARE', 'MONITOR', 'CMCS', 'SERV_NET' ]

    for t in start_tokens:
        n = s.find(f' {t} ')
        if n >= 0:
            # print('XXX', n)
            s = s[n+1:]
            s = re_pattern_core.sub('core', s)
            s = re_pattern_address.sub('address=X', s)
            s = re_pattern_unit.sub('unit=X', s)
            s = re_pattern_bit.sub('bit=X', s)
            s = clean_digits(s)
            #s = re_pattern_number.sub(' number ', s)
            return s
    return s

def load_data_and_labels_from_csv(filepath, clean_mode=0, header=None, int_labels=True, verbose=1):
    if verbose >= 2:
        print(f'Read csv file {filepath}')
    if isinstance(header, list) and len(header) == 0:
        flag_first_line = True
    else:
        if header is True:
            flag_first_line = True
            header = [] # csv file has column headers line, discard it
        else:
            flag_first_line = False
    with open(filepath) as f:
        data = []
        labels = []
        csv_f = csv.reader(f)
        for n, line in enumerate(csv_f):
            try:
                if flag_first_line:
                    header.append(line[0])
                    header.append(line[1])
                    flag_first_line = False
                else:
                    if clean_mode == 1:
                        data.append(clean_line(line[0]))
                    else:
                        data.append(line[0])
                    if int_labels:
                        labels.append(int(line[1]))
                    else:
                        labels.append(line[1])
            except:
                if verbose >= 1:
                    print(f'ERROR: invalid line {n}: {line}', file=sys.stderr)
    return data, labels


if __name__ == '__main__':
    verbose = 1
    clean_mode = 1
    header = []
    int_labels = True
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = '/data/bel/RUDN/data/1K/train.csv'
    data, labels = load_data_and_labels_from_csv(filepath, clean_mode=clean_mode, header=header, int_labels=int_labels, verbose=verbose)
    print(f'header: {header}', file=sys.stderr)
    for i in range(8):
        print(f'{labels[i]} : {data[i]}')
    print(f'Label class is {labels[i].__class__}')
