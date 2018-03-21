#!/usr/bin/env python

from __future__ import print_function
from __future__ import division
# from __future__ import absolute_import

import argparse
import codecs
import itertools
import json
import requests
import time

from os import path, makedirs
from subprocess import call


def ensure_dir(dir):
    try:
        makedirs(dir)
    except OSError:
        pass

def generate_all_tuples(proteins): # copied from iob_to_bind.json
    proteins = set(proteins)
    return [(a, b) for a, b in itertools.combinations(proteins, 2)]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_text', '-i', required=True, type=str)
    parser.add_argument('--output_json', '-o', required=True, type=str)
    parser.add_argument('--tmp_dir', '-t', default='tmp', type=str)
    # parser.add_argument('--classifier_model', '-c', required=True, type=str)
    args = parser.parse_args()

    basename = path.basename(args.input_text)
    
    ensure_dir(args.tmp_dir)  # not very necessary
    
    output = []
    positive_pairs = 0

    with codecs.open(args.input_text, 'r', encoding='utf-8') as input_file:
        for i, line in enumerate(input_file):
            line = line.strip()
            
            id, sentence = line.split('\t')
            
            # this should be doable from a single query...
            # TODO: understand how to parse fries
            while True:
                try:
                    response_index = requests.post('http://agathon.sista.arizona.edu:8080/odinweb/api/text', 
                                                   params={'text': sentence, 'output': 'indexcard'})
                    response_fries = requests.post('http://agathon.sista.arizona.edu:8080/odinweb/api/text', 
                                               params={'text': sentence, 'output': 'fries'})
                    break
                except Exception as e:
                    print("Exception. Trying one more time in 5 seconds")
                    print(e)
                    time.sleep(5)
                    pass
                
            try:
                data_indexcard = json.loads(response_index.content)
                data_fries = json.loads(response_fries.content)
                
                if 'cards' in data_indexcard:
                    pairs_from_json = [{
                        "participant_a": card['extracted_information']['participant_a']['entity_text'], 
                        "participant_b": card['extracted_information']['participant_b']['entity_text'], 
                        "label": 1, 
                        "interaction_type": "bind"
                    } for card in data_indexcard['cards']
                        if card['extracted_information']['interaction_type'] == 'binds']
                else:
                    pairs_from_json = []
                    
                if 'entities' in data_fries:
                    entities_from_json = [x['text'] for x in data_fries['entities']['frames']]
                else:
                    entities_from_json = []
                
                output.append({
                  "entities": entities_from_json, 
                  "extracted_information": pairs_from_json, 
                  "id": id, 
                  "text": sentence
                 })
                
                positive_pairs += len(pairs_from_json)
                with codecs.open(args.output_json, 'w', encoding='utf-8') as output_file:
                    json.dump(output, output_file, indent=True)
                    
            except Exception as e:
                print(e)
                print("ERROR: Skipping the sentence: {}".format(line))
                
            print("{} positive pairs found from {} sentences".format(positive_pairs, i+1))


    print("Saving the output to {}".format(args.output_json))
    with codecs.open(args.output_json, 'w', encoding='utf-8') as output_file:
        json.dump(output, output_file, indent=True)

    # TODO: remove tmp folder?


if __name__ == '__main__':
    main()
