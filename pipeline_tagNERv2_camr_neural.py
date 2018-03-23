#!/usr/bin/env python

from __future__ import print_function
from __future__ import division
# from __future__ import absolute_import

import codecs
import argparse

from os import path, makedirs
from subprocess import check_call


def ensure_dir(dir):
    try:
        makedirs(dir)
    except OSError:
        pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_text', '-i', required=True, type=str)
    parser.add_argument('--output_json', '-o', required=True, type=str)
    parser.add_argument('--tmp_dir', '-t', default='results/tmp', type=str)
    parser.add_argument('--classifier_model', '-c', default='', type=str)
    args = parser.parse_args()

    basename = path.basename(args.input_text)
    tokenized_input = path.join(args.tmp_dir, '{}.tokenized.txt'.format(basename))
    entities_output = path.join(args.tmp_dir, '{}.tokenized.txt.IOB'.format(basename))
    candidate_tuples_json = path.join(args.tmp_dir, '{}.candidates.json'.format(basename))
    
    ensure_dir(args.tmp_dir)  # not very necessary
    
    print("Tokenizing...")
    # print("Adding spaces around -")
    with codecs.open(args.input_text, encoding='utf-8') as fr:
        with codecs.open(tokenized_input, 'w', encoding='utf-8') as fw:
            for line in fr.readlines():
                id, sentence = line[:-1].split('\t')  # \n symbol
                #sentence = sentence.replace('-',' - ')
                sentence = ' '.join(sentence.split())
                fw.write("{}\t{}\n".format(id, sentence))

    print('Running NER...')
    check_call(['bash', 'tag_NER.sh',
                '-i', path.join('../..', tokenized_input), # because cwd gets two levels below
                '-f', 'IOB'], cwd='submodules/tag_NER_v2')
    print('Done\n')
    # the output is entities_output
    
    
    print('Building interaction tuples with unknown labels...')
    check_call(['python', 'iob_to_bind_json.py',
                '--input_text', args.input_text,
                '--input_iob2', entities_output,
                '--output_json', args.output_json]) #candidate_tuples_json])
    print('Done\n')
    
    print('Adding AMRs')
    check_call(['python3', 'add_amr.py',
                '--input_text', args.input_text,
                '--input_json', args.output_json,
                '--model', 'amr2_bio7_best_after_2_fscore_0.6118.m',
                '--output_json', args.output_json,
                '--tmp_dir', args.tmp_dir])
    print('Done\n')
    
    print('Extracting AMR paths')
    check_call(['python', 'append_amr_paths.py',
                '--input_json', args.output_json,
                '--output_json', args.output_json,
                '--tmp_dir', args.tmp_dir])
    print('Done\n')
    
    

    # add command to generate SDG .. paths
    
    raise Exception("Classifier is not ready!")

    print('Detecting true interactions...')
    check_call(['python',
                'submodules/RelationClassification/inference.py',
                '--input_json', candidate_tuples_json,
                '--output_json', args.output_json,
                '--model_path', args.classifier_model
          ])
    print('Done\n')

    # replace protein names with identifiers

    # TODO: remove tmp folder?


if __name__ == '__main__':
    main()
