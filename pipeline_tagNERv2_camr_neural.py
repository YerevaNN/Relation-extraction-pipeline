#!/usr/bin/env python

from __future__ import print_function
from __future__ import division
# from __future__ import absolute_import

import io
import argparse
import json

import os
from subprocess import check_call


def ensure_dir(dir):
    try:
        os.makedirs(dir)
    except OSError:
        pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_text', '-i', required=True, type=str)
    parser.add_argument('--output_json', '-o', required=True, type=str)
    parser.add_argument('--tmp_dir', '-t', default='results/tmp', type=str)
    parser.add_argument('--classifier_model', '-c', default='', type=str)
    parser.add_argument('--use_amr', '-uamr', action='store_true')
    parser.add_argument('--use_sdg', '-usdg', action='store_true')
    parser.add_argument('--use_ground_truth_entities', '-ug', action='store_true')
    parser.add_argument('--ground_truth_json', '-g', type=str)
    args = parser.parse_args()

    basename = os.path.basename(args.input_text)
    args.tmp_dir = os.path.abspath(args.tmp_dir)
    print("Using tmp dir: {}".format(args.tmp_dir))
    
    ensure_dir(args.tmp_dir)  # not very necessary
    
    if args.use_ground_truth_entities:
        print("Getting ground truth entities and pairs...")
        
        with io.open(args.ground_truth_json, encoding='utf-8') as fr:
            with io.open(args.output_json, 'w', encoding='utf-8') as fw:
                ground_truth = fr.read()
                fw.write(ground_truth)
        
        print('Done\n')
        
    else:
        tokenized_input = os.path.join(args.tmp_dir, '{}.tokenized.txt'.format(basename))
        entities_output = os.path.join(args.tmp_dir, '{}.tokenized.txt.IOB'.format(basename))
        candidate_tuples_json = os.path.join(args.tmp_dir, '{}.candidates.json'.format(basename))

        print("Tokenizing...")
        # print("Adding spaces around -")
        with io.open(args.input_text, encoding='utf-8') as fr:
            with io.open(tokenized_input, 'w', encoding='utf-8') as fw:
                for line in fr.readlines():
                    id, sentence = line[:-1].split('\t')  # \n symbol
                    #sentence = sentence.replace('-',' - ')
                    sentence = ' '.join(sentence.split())
                    fw.write("{}\t{}\n".format(id, sentence))

        print('Running NER...')
        check_call(['bash', 'tag_NER.sh',
                    '-i', tokenized_input,
                    '-f', 'IOB'], cwd='submodules/tag_NER_v2')
        print('Done\n')
        # the output is entities_output

        print('Building interaction tuples with unknown labels...')
        check_call(['python', 'iob_to_bind_json.py',
                    '--input_text', args.input_text,
                    '--input_iob2', entities_output,
                    '--output_json', args.output_json]) #candidate_tuples_json])
        print('Done\n')

    
    if args.use_amr:
        print('Adding AMRs...')
        check_call(['python3', 'add_amr.py',
                    '--input_text', args.input_text,
                    '--input_json', args.output_json,
                    '--model', 'amr2_bio7_best_after_2_fscore_0.6118.m',
                    '--output_json', args.output_json,
                    '--tmp_dir', args.tmp_dir])
        print('Done\n')
        
        print('Extracting AMR paths...')
        check_call(['python3', 'append_amr_paths.py',
                    '--input_json', args.output_json,
                    '--output_json', args.output_json,
                    '--tmp_dir', args.tmp_dir])
        print('Done\n')

    if args.use_sdg:        
        print('Adding Stanford Dependency Graphs...')
        check_call(['python', 'add_sdg.py',
                    '--input_text', args.input_text,
                    '--input_json', args.output_json,
                    '--output_json', args.output_json,
                    '--tmp_dir', args.tmp_dir])
        print('Done\n')

        print('Extracting SDG paths...')
        check_call(['python', 'append_sdg_paths.py',
                    '--input_json', args.output_json,
                    '--output_json', args.output_json])
        print('Done\n')
    

    # raise Exception("Classifier is not ready!")
    
    before_classifier = os.path.join(args.tmp_dir, '{}.before-classifier.json'.format(basename))
    after_classifier = os.path.join(args.tmp_dir, '{}.after-classifier.json'.format(basename))
    
    print("Converting dense JSON to flat JSON: {} ...".format(before_classifier))      
    with io.open(args.output_json, encoding='utf-8') as fr:
        with io.open(before_classifier, 'w', encoding='utf-8') as fw:
            dense = json.load(fr)
            flat = {}
            for sentence in dense:
                for i, pair in enumerate(sentence['extracted_information']):
                    id = "{}|{}".format(sentence['id'], i)
                    sentence['extracted_information'][i]['id'] = id
                    flat[id] = {
                        'text': sentence['text'],
                        'interaction_tuple': [
                            pair['interaction_type'],
                            '',
                            pair['participant_a'],
                            pair['participant_b']
                        ],
                        'label': 1 if pair['label'] != 0 else 0  # TODO: -1s
                    }
                    if 'amr_path' in pair: 
                        flat[id]['amr_path'] = pair['amr_path']
                    if 'sdg_path' in pair: 
                        flat[id]['sdg_path'] = pair['sdg_path']
            
            flat_json_string = json.dumps(flat, indent=True)
            fw.write(flat_json_string)
    print("Done!")

    print('Detecting true interactions...')
    check_call(['python',
                'predict.py',
                '--input_path', before_classifier,
                '--output_path', after_classifier,
                '--model_path', args.classifier_model
          ], cwd='submodules/RelationClassification/')
    print('Done\n')

    print("Reading classifier output from flat JSON: {} ...".format(after_classifier))      
    with io.open(after_classifier, encoding='utf-8') as fr:
        with io.open(args.output_json, 'w', encoding='utf-8') as fw:
            flat = json.load(fr)
            for sentence in dense:
                for pair in sentence['extracted_information']:
                    pair['label'] = flat[pair['id']]['prediction']
                    
            dense_json_string = json.dumps(dense, indent=True)
            fw.write(dense_json_string)
    print("Done!")

    # replace protein names with identifiers

    # TODO: remove tmp folder?


if __name__ == '__main__':
    main()
