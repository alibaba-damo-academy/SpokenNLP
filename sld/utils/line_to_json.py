#!/usr/bin/python
import os
import json

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_file_text')
    parser.add_argument('--input_file_idx')
    parser.add_argument('--output_file')

    args = parser.parse_args()
    print(args)

    with open(args.input_file_text, 'r') as fi1:
        with open(args.input_file_idx, 'r') as fi2:
            with open(args.output_file, 'w') as fw:
                for line1, line2 in zip(fi1, fi2):
                    text = line1.strip()
                    idx = [int(x) for x in line2.strip().split()]
                    json_string = json.dumps({'text': text.lower(), 'idx': idx})
                    fw.write(json_string + '\n')
