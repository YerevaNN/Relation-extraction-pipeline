#!/usr/bin/env python

from __future__ import print_function
from __future__ import division
# from __future__ import absolute_import

import io
import argparse
from glob import glob
import html
import json
import numpy as np
from nltk.metrics.distance import edit_distance
import re

import os
from subprocess import check_call, check_output, run


def ensure_dir(dir):
    try:
        os.makedirs(dir)
    except OSError:
        pass


def double_normalize_text(s, lower=True):
    s = html.unescape(s)
    s = re.sub(r'([\.,\'\"_\(\)/-])', r' \1 ', s)
    s = re.sub('\s+', ' ', s)
    if lower:
        s = s.lower()
    s = s.strip()
    return s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_text', '-i', required=True, type=str)
    parser.add_argument('--output_json', '-o', required=True, type=str)
    parser.add_argument('--tmp_dir', '-t', default='results/tmp', type=str)
    parser.add_argument('--classifier_type', '-ct', default='RelationClassification', type=str,
                        choices=['RelationClassification', 'fasttext'])
    parser.add_argument('--classifier_model', '-c', nargs='*', type=str)
    parser.add_argument('--classifier_preprocessor', '-cp', nargs='*', type=str)
    parser.add_argument('--use_amr', '-uamr', action='store_true')
    parser.add_argument('--amrs_from', type=str)
    parser.add_argument('--tokenize', action='store_true')
    parser.add_argument('--use_sdg', '-usdg', action='store_true')
    parser.add_argument('--sdg_model', '-sdg', default='stanford', type=str,
                        choices=['stanford', 'spacy'])
    parser.add_argument('--entity_recognizer', '-er', default='None', type=str, 
                        choices=['None', 'tagNERv2', 'tagNERv3', 'byteNER'])
    parser.add_argument('--entities_from', type=str)
    parser.add_argument('--anonymize', '-a', action='store_true')
    parser.add_argument('--add_symmetric_pairs', '-sym', action='store_true')
    parser.add_argument('--ensembling_mode', '-ens', type=str, default='average',
                        choices=['average', 'majority_vote'])
    args = parser.parse_args()

    basename = os.path.basename(args.input_text)
    args.tmp_dir = os.path.abspath(args.tmp_dir)
    print("Using tmp dir: {}".format(args.tmp_dir))
    
    ensure_dir(args.tmp_dir)  # not very necessary

    
    if args.entity_recognizer == 'None':
        if args.entities_from:
            print("Getting ground truth entities and pairs...")

            with io.open(args.entities_from, encoding='utf-8') as fr:
                with io.open(args.output_json, 'w', encoding='utf-8') as fw:
                    ground_truth = fr.read()
                    fw.write(ground_truth)

            print('Done\n')
        
        else:
            raise Exception("--entities_from is not specified")

    elif args.entity_recognizer == 'tagNERv2':
        tokenized_input = os.path.join(args.tmp_dir, '{}.tokenized.txt'.format(basename))
        entities_output = os.path.join(args.tmp_dir, '{}.tokenized.txt.IOB'.format(basename))
        # candidate_tuples_json = os.path.join(args.tmp_dir, '{}.candidates.json'.format(basename))

        print("Tokenizing...")
        # print("Adding spaces around -")
        with io.open(args.input_text, encoding='utf-8') as fr:
            with io.open(tokenized_input, 'w', encoding='utf-8') as fw:
                for line in fr.readlines():
                    id, sentence = line[:-1].split('\t')  # \n symbol
                    #sentence = sentence.replace('-',' - ')
                    sentence = ' '.join(sentence.split())
                    fw.write("{}\t{}\n".format(id, sentence))

        print('Running tagNERv2...')
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
        
    elif args.entity_recognizer == 'tagNERv3':
        
        tokenized_input = os.path.join(args.tmp_dir, '{}.tokenized.txt'.format(basename))
        entities_output = os.path.join(args.tmp_dir, '{}.tokenized.txt.IOB'.format(basename))
        # candidate_tuples_json = os.path.join(args.tmp_dir, '{}.candidates.json'.format(basename))

        print("Tokenizing...")
        # print("Adding spaces around -")
        with io.open(args.input_text, encoding='utf-8') as fr:
            with io.open(tokenized_input, 'w', encoding='utf-8') as fw:
                for line in fr.readlines():
                    id, sentence = line[:-1].split('\t')  # \n symbol
                    #sentence = sentence.replace('-',' - ')
                    sentence = ' '.join(sentence.split())
                    fw.write("{}\t{}\n".format(id, sentence))

        print('Running tagNERv3 inside Docker...')

        with open(tokenized_input, 'r') as f_in, open(entities_output, 'w') as f_out:
            p = run(['nvidia-docker', 'run',
                                '-i', '--rm',
                                'yerevann/tag-ner-v3',
                                '-i', '/dev/stdin',
                                '-f', 'IOB'],
                               stdin=f_in, stdout=f_out)
            
        print('Done\n')
        # the output is entities_output
        print('Building interaction tuples with unknown labels...')
        check_call(['python', 'iob_to_bind_json.py',
                    '--input_text', args.input_text,
                    '--input_iob2', entities_output,
                    '--output_json', args.output_json]) #candidate_tuples_json])
        print('Done\n')
        
    elif args.entity_recognizer == 'byteNER':
        input_without_ids = os.path.join(args.tmp_dir, '{}.noids.txt'.format(basename))
        entities_output = os.path.join(args.tmp_dir, '{}.IOB'.format(basename))
        entities_output_chr = os.path.join(args.tmp_dir, '{}.IOB.chr'.format(basename))
        
        print('Removing IDs from input for byteNER')
        with io.open(args.input_text, encoding='utf-8') as fr:
            with io.open(input_without_ids, 'w', encoding='utf-8') as fw:
                for line in fr.readlines():
                    id, sentence = line[:-1].split('\t')  # \n symbol
                    fw.write("{}\n".format(sentence))
        print("Done")
        
        print('Running byteNER...')
        # requires Keras 2.0.6 on python2!
        env = os.environ.copy()
        env['KERAS_BACKEND'] = 'theano'
        env['THEANO_FLAGS'] = 'dnn.enabled=False'
        check_call(['python2', 'tagger.py',
                    '-m', 'models/20CNN,dropout0.5,bytedrop0.3,lr0.0001,bytes,bpe,blstm,crf,biocreative.model', 
                    '-i', input_without_ids, 
                    '-o', entities_output,
                    '--output_format', 'iob'], 
                   cwd='submodules/byteNER', 
                   env=env)
        print('Done\n')        
        
        print('Building interaction tuples with unknown labels...')
        check_call(['python', 'iob_to_bind_json.py',
                    '--character_level',
                    '--input_text', args.input_text,
                    '--input_iob2', entities_output_chr,
                    '--output_json', args.output_json]) #candidate_tuples_json])
        print('Done\n')
        
    pretokenized_input = os.path.join(args.tmp_dir, '{}.pretokenized.txt'.format(basename))
    if args.tokenize:
        with open(args.input_text, 'r', encoding='utf-8') as fr:
            with open(pretokenized_input, 'w', encoding='utf-8') as fw:
                for line in fr:
                    id, sentence = line[:-1].split('\t')
                    sentence = double_normalize_text(sentence, lower=False)
                    # although fasttext vectors require lower(), RelClass handles it internally
                    fw.write("{}\t{}\n".format(id, sentence))
    else:
        with open(args.input_text, 'r', encoding='utf-8') as fr:
            with open(pretokenized_input, 'w', encoding='utf-8') as fw:
                fw.write(fr.read())     
       
    if args.add_symmetric_pairs:
        # useful for symmetric interactions like `bind`
        with io.open(args.output_json, 'r', encoding='utf-8') as f:
            dense = json.load(f)
            print("Adding symmetric pairs...")
            for sentence in dense:
                sym = []
                for i, pair in enumerate(sentence['extracted_information']):
                    reverse_pair = pair.copy()
                    reverse_pair['participant_a'] = pair['participant_b']
                    reverse_pair['participant_b'] = pair['participant_a']
                    reverse_pair['_sym_of'] = i
                    sym.append(reverse_pair)
                sentence['extracted_information'] += sym
        with io.open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump(dense, f)
                
    
    if args.use_amr:
        print('Adding AMRs...')
        if args.amrs_from:
            with open(args.amrs_from, 'r', encoding='utf-8') as f:
                amrs = json.load(f)
            amr_dict = {}
            for sample in amrs:
                amr_dict[sample['id']] = sample['amr']
            with open(args.output_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for sentence in data:
                sentence['amr'] = amr_dict[sentence['id']]
            with open(args.output_json, 'w', encoding='utf-8') as f:
                json.dump(data, f)
        else:
            check_call(['python3', 'add_amr.py',
                        '--input_text', pretokenized_input,
                        '--input_json', args.output_json,
                        '--model', 'amr2_bio7_best_after_2_fscore_0.6118.m',
                        #'--model', 'bio_model_best.m',
                        '--output_json', args.output_json,
                        '--tmp_dir', args.tmp_dir])
        print('Done\n')
        
        print('Extracting AMR paths...')
        check_call(['python3', 'append_amr_paths.py',
                    '--input_json', args.output_json,
                    '--output_json', args.output_json,
                    '--tmp_dir', args.tmp_dir])
        print('Done\n')

        print('Appending Amr Soft-Matching Statistics...')
        with open(args.output_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for sentence in data:
            for info in sentence['extracted_information']:
                participant_a = info['participant_a']
                participant_b = info['participant_b']
                if not info['amr_path']:
                    info['amr_path'] = '{} _nopath_ {}'.format(participant_a,
                                                               participant_b)
                    info['amr_soft_match_distance_a'] = -1 
                    info['amr_soft_match_distance_b'] = -1 
                else:
                    amr_match_a = info['amr_path'].split()[0]
                    amr_match_b = info['amr_path'].split()[-1]
                    info['amr_soft_match_distance_a'] = edit_distance(participant_a,
                                                                      amr_match_a)
                    info['amr_soft_match_distance_b'] = edit_distance(participant_b,
                                                                      amr_match_b)
        with open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump(data, f)
        print('Done')

    if args.use_sdg:        
        print('Adding Stanford Dependency Graphs...')
        check_call(['python', 'add_sdg.py',
                    '--input_text', pretokenized_input,
                    '--input_json', args.output_json,
                    '--output_json', args.output_json,
                    '--model', args.sdg_model,
                    '--tmp_dir', args.tmp_dir])
        print('Done\n')

        print('Extracting SDG paths...')
        check_call(['python', 'append_sdg_paths.py',
                    '--input_json', args.output_json,
                    '--output_json', args.output_json])
        print('Done\n')
    
        print('Appending SDG Soft-Matching Statistics...')
        with open(args.output_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for sentence in data:
            for info in sentence['extracted_information']:
                 participant_a = info['participant_a']
                 participant_b = info['participant_b']
                 if not info['sdg_path']:
                    info['sdg_path'] = '{} _nopath_ {}'.format(participant_a,
                                                               participant_b)
                    info['sdg_soft_match_distance_a'] = -1 
                    info['sdg_soft_match_distance_b'] = -1 
                 else:
                    sdg_match_a = info['sdg_path'].split()[0]
                    sdg_match_b = info['sdg_path'].split()[-1]
                    info['sdg_soft_match_distance_a'] = edit_distance(participant_a,
                                                                      sdg_match_a)
                    info['sdg_soft_match_distance_b'] = edit_distance(participant_b,
                                                                      sdg_match_b)
        with open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump(data, f)

        print('Done')

    # raise Exception("Classifier is not ready!")
    
    before_classifier = os.path.join(args.tmp_dir, '{}.before-classifier.json'.format(basename))
    after_classifier = os.path.join(args.tmp_dir, '{}.after-classifier.0.json'.format(basename))
    after_classifier_format_string = os.path.join(args.tmp_dir,
                                                  '{}.after-classifier.{}.json'.format(basename, "{}"))
    
    print("Converting dense JSON to flat JSON: {} ...".format(before_classifier))      
    with io.open(args.output_json, encoding='utf-8') as fr:
        dense = json.load(fr)
        flat = {}
        if args.anonymize:
            print("Anonymizing...")
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
                if '_sym_of' in pair:
                    flat[id]['_sym_of'] = "{}|{}".format(sentence['id'], pair['_sym_of'])
                    
                tokenized_text = None
                if args.anonymize:
                    placeholder_a = '__participant_a__'
                    placeholder_b = '__participant_b__'

                    flat[id]['interaction_tuple'][2] = placeholder_a
                    flat[id]['interaction_tuple'][3] = placeholder_b
                    if not args.use_sdg and not args.use_amr:
                        raise NotImplementedError('Anonymization for this \
                                                   setting is not implemented')
                    if args.use_sdg:
                        sdg_match_a = pair['sdg_path'].split()[0]
                        sdg_match_b = pair['sdg_path'].split()[-1]
                        pair['sdg_path'] = pair['sdg_path'].replace(sdg_match_a,
                                                                    placeholder_a)
                        pair['sdg_path'] = pair['sdg_path'].replace(sdg_match_b,
                                                                    placeholder_b)
                        tokenized_text = sentence['tokenized_text']
                        tokenized_text = [placeholder_a if word == sdg_match_a
                                          else word for word in tokenized_text]
                        tokenized_text = [placeholder_b if word == sdg_match_b
                                          else word for word in tokenized_text]
                        
                        # sdg = sentence['sdg'].replace(sdg_match_a,
                        #                               placeholder_a)
                        # sdg = sdg.replace(sdg_match_b,
                        #                   placeholder_b)

                    if args.use_amr:
                        amr_match_a = pair['amr_path'].split()[0]
                        amr_match_b = pair['amr_path'].split()[-1]
                        pair['amr_path'] = pair['amr_path'].replace(amr_match_a,
                                                                    placeholder_a)
                        pair['amr_path'] = pair['amr_path'].replace(amr_match_b,
                                                                    placeholder_b)

                    if args.use_sdg:
                        participant_a = sdg_match_a
                        participant_b = sdg_match_b
                    else:
                        participant_a = amr_match_a
                        participant_b = amr_match_b
                    text = flat[id]['text']
                    text = text.replace(participant_a, placeholder_a)
                    text = text.replace(participant_b, placeholder_b)
                    flat[id]['text'] = text

                if 'amr_path' in pair:
                    flat[id]['amr_path'] = pair['amr_path']
                if 'sdg_path' in pair:
                    flat[id]['sdg_path'] = pair['sdg_path']
                if 'tokenized_text' in sentence:
                    if tokenized_text is not None:
                        # custom, anonymized version
                        flat[id]['tokenized_text'] = tokenized_text
                    else:
                        # general version
                        flat[id]['tokenized_text'] = sentence['tokenized_text']
                if 'pos_tags' in sentence:
                    flat[id]['pos_tags'] = sentence['pos_tags']


    flat_json_string = json.dumps(flat, indent=True)
    
    with io.open(before_classifier, 'w', encoding='utf-8') as fw:
        fw.write(flat_json_string)
    print("Done!")

    print('Detecting true interactions using {} ...'.format(args.classifier_type))
    if args.classifier_type == "RelationClassification":
        for i, (model, processor) in enumerate(zip(args.classifier_model,
                                                   args.classifier_preprocessor)):
            print('Running model number {}'.format(i))
            print('Model filepath: {}'.format(model))
            check_call(['python2',
                        'predict.py',
                        '--input_path', before_classifier,
                        '--output_path', after_classifier_format_string.format(i),
                        '--processor_path', processor,
                        '--model_path', model,
                  ], cwd='submodules/RelationClassification/')
    elif args.classifier_type == "fasttext":
        # TODO: this does not support multiple models!
        # TODO: this is pretty ugly. 
        # Preprocessing and postprocessing for fasttext and RelClass should be at the same level
        before_fasttext = os.path.join(args.tmp_dir, '{}.before-fasttext.txt'.format(basename))
        fasttext_keys = []
        
        with io.open(before_fasttext, 'w', encoding='utf-8') as fw:
            for k,v in flat.items():
                fw.write("{}\n".format(v['text']))
                fasttext_keys.append(k)
                
        fasttext_output = check_output(['fasttext',
            'predict',
            args.classifier_model,
            before_fasttext,
            #after_classifier
           ])
        fasttext_labels = fasttext_output.decode('utf-8').split('\n')
        
        for i, k in enumerate(fasttext_keys):
            label_string = fasttext_labels[i]
            if not label_string.startswith("__label__"):
                print("Error: invalid label: {}".format(label_string))
            else:
                flat[k]['prediction'] = int(label_string[9:])
                
        flat_json_string = json.dumps(flat, indent=True)
        with io.open(after_classifier, 'w', encoding='utf-8') as fw:
            fw.write(flat_json_string)

    for after_classifier in sorted(glob(after_classifier_format_string.format('*'))):
        print("Reading classifier output from flat JSON: {} ...".format(after_classifier))
        with io.open(after_classifier, encoding='utf-8') as fr:
            flat = json.load(fr)
            found = 0
            missing = 0
            for sentence in dense:
                for pair in sentence['extracted_information']:
                    if pair['id'] in flat:
                        if 'predictions' not in pair:
                            pair['predictions'] = []
                        if 'probabilities' not in pair:
                            pair['probabilities'] = []
                        pair['predictions'].append(flat[pair['id']]['prediction'])
                        if 'probabilities' in flat[pair['id']]:
                            pair['probabilities'].append(flat[pair['id']]['probabilities'])
                        found += 1
                    else:
                        missing += 1
            print("{}/{} items did not have predictions in {}".format(missing,
                                                                      missing+found,
                                                                      after_classifier))
    

    #  Performing Ensembling

    for sentence in dense:
        for pair in sentence['extracted_information']:
            if args.ensembling_mode == 'majority_vote':
                if sum(pair['predictions']) / len(pair['predictions']) < 0.5:
                    pair['label'] = 0
                else:
                    pair['label'] = 1
            else: # args.ensembling_mode = 'average'
                prob = np.array(pair['probabilities']).mean(axis=0)
                pair['label'] = int(prob.argmax())



    with io.open(args.output_json, 'w', encoding='utf-8') as fw:
        dense_json_string = json.dumps(dense, indent=True)
        fw.write(dense_json_string)
    print("Done!")

    # replace protein names with identifiers

    # TODO: remove tmp folder?


if __name__ == '__main__':
    main()
