#!/usr/bin/env python

from __future__ import division
from __future__ import print_function
# from __future__ import absolute_import

import codecs
import argparse
import json
import itertools


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_text', '-it', required=True, type=str)
    parser.add_argument('--input_iob2', '-i', required=True, type=str)
    parser.add_argument('--output_json', '-o', required=True, type=str)
    parser.add_argument('--clean_entities', action='store_true')
    args = parser.parse_args()
    
    protein_labels = ['proteingene', 'smallmolecule']  #, 'cellularcomponent', 'celltypeline']
    
    print("Read {input_iob2:} IOB2 file and extract entities marked as '{entities:}', store the results to {output_json:}, use sentences from {input_text:}".format(
        input_iob2=args.input_iob2,
        output_json=args.output_json,
        entities="' or '".join(protein_labels),
        input_text=args.input_text))

    
    def clean(w):
        # TODO: replace this with better tokenizer?!
        if w[-1] in [',', '.', ')', ']', '?', '!']:
            w = w[:-1]
        return w

    def generate_all_tuples(proteins):
        if args.clean_entities:
            proteins = [clean(p) for p in proteins]
        proteins = set(proteins)
        return [{
            "participant_a": a, 
            "participant_b": b, 
            "label": 0, 
            "interaction_type": "bind"
           } for a, b in itertools.combinations(proteins, 2)]

    curr_proteins = []
    curr_word = ''
    passage = ''
    output = []
    bind = False
    
    sentences = {}
    with codecs.open(args.input_text, encoding='utf-8') as file:
        for line in file:
            id, sentence = line.split('\t')
            sentences[id] = sentence

    with codecs.open(args.input_iob2, encoding='utf-8') as file:
        for line in file:
            if line == '\n':
                if curr_word:
                    curr_proteins.append(curr_word)
                    curr_word = ''
                if len(curr_proteins) > 1: 
                    all_tuples = generate_all_tuples(curr_proteins)
                else:
                    all_tuples = []
                
                output.append({
                    "entities": curr_proteins, 
                    "extracted_information": all_tuples, 
                    "id": id, 
                    "text": sentences[id]
                })
                    
                #bind = False
                curr_proteins = []
                passage = ''
                continue

            try:
                id, _, _, word, label = line.split()
            except:
                print("ERROR")
                print(line)

            passage += ' '
            passage += word

#            if word in ['bind', 'binding', 'bound', 'binds']:
#                bind = True
            t = label[0]
            if t == 'O':
                if curr_word:
                    curr_proteins.append(curr_word)
                    curr_word = ''
                continue
            label = label[2:]
            if t == 'I' and curr_word:
                curr_word += ' ' + word
            else:
                if curr_word:
                    curr_proteins.append(curr_word)
                    curr_word = ''
                if label in protein_labels:
                    curr_word = word

    with codecs.open(args.output_json, 'w', encoding='utf-8') as fd:
        json.dump(output, fd, indent=True)


if __name__ == '__main__':
    main()
