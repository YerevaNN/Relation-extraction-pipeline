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
from tqdm import tqdm

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
        for i, line in tqdm(enumerate(input_file)):
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

                if 'entities' in data_fries:
                    entity_mentions = [{
                        "name": x['text'],
                        "label": x['type'],
                        "mention": [x['start-pos']['offset'], x['end-pos']['offset'] - 1], # -1 is important for END
                        "grounding": x['xrefs'][0]['namespace'] + ':' + x['xrefs'][0]['id']
                    } for x in data_fries['entities']['frames']]
                else:
                    entity_mentions = []

                entities = {}
                for mention in entity_mentions:
                    if mention['name'] not in entities:
                        entities[mention['name']] = {
                            "labels_major": set(),
                            "mentions": [],
                            "exists": True,
                            "grounding": set()
                        }
                    entities[mention['name']]["labels_major"].add(mention['label'])
                    entities[mention['name']]["mentions"].append(mention['mention'])
                    entities[mention['name']]["grounding"].add(mention['grounding'])

                # convert set to list
                for e, e_obj in entities.items():
                    e_obj["labels_major"] = list(e_obj["labels_major"])
                    e_obj["grounding"] = list(e_obj["grounding"])
                    if len(e_obj["grounding"]) > 1:
                        print("\nMultiple groundings for the same entity {} (ID={})".format(e, id))
                    e_obj["grounding"] = e_obj["grounding"][0]

                unique_entities = {i: {
                    "versions": {e:e_obj},
                    "labels_major": e_obj['labels_major'],
                    "all_implicit": False,
                    "grounding": e_obj["grounding"]
                } for i, (e, e_obj) in enumerate(entities.items())}

                pairs_from_json = []
                if 'cards' in data_indexcard:
                    for card in data_indexcard['cards']:
                        if card['extracted_information']['interaction_type'] != 'binds':
                            continue
                        a_grounding = card['extracted_information']['participant_a']['identifier']
                        b_grounding = card['extracted_information']['participant_b']['identifier']
                        a_ids = [ue for ue, ue_obj in unique_entities.items() if ue_obj['grounding'] == a_grounding]
                        b_ids = [ue for ue, ue_obj in unique_entities.items() if ue_obj['grounding'] == b_grounding]
                        if len(a_ids) != 1:
                            print(u"Participant A {}: {} ids found (ID={})".format(
                                card['extracted_information']['participant_a']['entity_text'],
                                len(a_ids),
                                id
                            ))
                        if len(b_ids) != 1:
                            print(u"Participant B {}: {} ids found (ID={})".format(
                                card['extracted_information']['participant_b']['entity_text'],
                                len(a_ids),
                                id
                            ))
                        pairs_from_json.append({
                            "participant_a": card['extracted_information']['participant_a']['entity_text'],
                            "participant_b": card['extracted_information']['participant_b']['entity_text'],
                            "participant_ids": [a_ids[0], b_ids[0]],
                            "label": 1,
                            "interaction_type": "bind"
                        })

                output.append({
                    "entities": entities,
                    "unique_entities": unique_entities,
                    "extracted_information": pairs_from_json,
                    "id": id,
                    "text": sentence
                 })
                
                positive_pairs += len(pairs_from_json)
                with codecs.open(args.output_json, 'w', encoding='utf-8') as output_file:
                    json.dump(output, output_file, indent=True)
                    
            except Exception as e:
                print(e)
                print(u"ERROR: Skipping the sentence: {}".format(line))
                
            #print("{} positive pairs found from {} sentences".format(positive_pairs, i+1))


    print("Saving the output to {}".format(args.output_json))
    with codecs.open(args.output_json, 'w', encoding='utf-8') as output_file:
        json.dump(output, output_file, indent=True)

    # TODO: remove tmp folder?


if __name__ == '__main__':
    main()
