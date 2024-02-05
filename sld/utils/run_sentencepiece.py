#!/usr/bin/python
import os
import json
import sentencepiece as spm
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_file_train')
    parser.add_argument('--input_file_validation')
    parser.add_argument('--input_file_test')
    parser.add_argument('--output_file_train')
    parser.add_argument('--output_file_validation')
    parser.add_argument('--output_file_test')
    parser.add_argument('--model_prefix')
    parser.add_argument('--vocab_size')

    args = parser.parse_args()
    print(args)


    def number_to_char(x, offset=0):
        return chr(x + 256 + offset)


    with open(args.input_file_train, 'r') as fi:
        with open(args.input_file_train + '.line', 'w') as fw:
            for line in fi:
                json_data = json.loads(line.strip())
                target = json_data['idx']
                target = ''.join([number_to_char(x) for x in target])
                fw.write(target + '\n')

    spm.SentencePieceTrainer.train(input=args.input_file_train + '.line', model_prefix=args.model_prefix,
                                   vocab_size=args.vocab_size, model_type='unigram', normalization_rule_name='identity',
                                   remove_extra_whitespaces=False)

    sp = spm.SentencePieceProcessor(model_file=args.model_prefix + '.model')

    file_list = [(args.input_file_train, args.output_file_train),
                 (args.input_file_validation, args.output_file_validation),
                 (args.input_file_test, args.output_file_test)]

    for file_in, file_out in file_list:
        with open(file_in, 'r') as fi:
            with open(file_out, 'w') as fw:
                for line in fi:
                    json_data = json.loads(line.strip())
                    target = json_data['idx']
                    target = ''.join([number_to_char(x) for x in target])
                    subword_idx = sp.encode(target, out_type=int)
                    subword = sp.encode(target, out_type=str)
                    json_data['subword'] = ' '.join([str(x) for x in subword])
                    json_data['subword_idx'] = subword_idx
                    json_string = json.dumps(json_data)
                    fw.write(json_string + '\n')
