from __future__ import print_function
from __future__ import division

import io
import argparse
import json
import os
from subprocess import check_call
from tqdm import tqdm


def sdg_id(sdg):
    return sdg.split('\n')[0][7:]  # Removing '# ::id '


def remove_sdg_id(sdg):
    return '\n'.join(sdg.split('\n')[1:]) # Removing the ID line


def parse_sdg_line(sdg_line):
    line_arr = sdg_line.split('\t')
    id = int(line_arr[0])
    word = line_arr[1]
    lemma = line_arr[2]
    pos = line_arr[3]
    parent_id = int(line_arr[4])
    edge = line_arr[5]
    return id, word, lemma, pos, parent_id, edge


def append_tokenized_text(sample):
    sdg = sample['sdg']
    tokenized_text = []
    pos_tags = []
    for line in sdg.split('\n'):
        _, word, _, pos, _, _ = parse_sdg_line(line)
        tokenized_text.append(word)
        pos_tags.append(pos)
    sample['tokenized_text'] = tokenized_text
    sample['pos_tags'] = pos_tags
    return sample

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_text', '-it', required=True, type=str)
    parser.add_argument('--input_json', '-ij', required=True, type=str)
    parser.add_argument('--tmp_dir', '-t', required=True, type=str)
    parser.add_argument('--output_json', '-o', required=True, type=str)
    parser.add_argument('--model', '-m', default='stanford', type=str)
    args = parser.parse_args()
    
    sdg_output = os.path.join(args.tmp_dir, 'output.conll')

    if args.model == 'stanford':
        check_call(['bash', 'submodules/conll_parser/parse_script.sh',
                    args.input_text, sdg_output])
    elif args.model == 'spacy':
        check_call(['python3', 'submodules/spacy_wrapper/conll/parser.py',
                    '--input', args.input_text, 
                    '--output', sdg_output])
        
    with io.open(sdg_output, 'r', encoding='utf-8') as f:
        sdgs = f.read().split('\n\n')
       
    if sdgs[-1].strip():
        sdgs = sdgs[:-1]

    sdg_dict = {sdg_id(sdg) : remove_sdg_id(sdg) for sdg in sdgs}

    with io.open(args.input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for i in tqdm(range(len(data))):
        id = data[i]['id']
        if id not in sdg_dict:
            print("Sentence with ID='{}' has no SDG graph".format(id))
            continue
        
        data[i]['sdg'] = sdg_dict[id]
        data[i] = append_tokenized_text(data[i])


    with io.open(args.output_json, 'w', encoding='utf-8') as f:
        dumps = json.dumps(data, indent=True)
        f.write(dumps)

if __name__ == '__main__':
    main()
