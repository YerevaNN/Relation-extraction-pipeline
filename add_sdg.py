from __future__ import print_function
from __future__ import division

import io
import argparse
import json
import os
from subprocess import check_call


def sdg_id(sdg):
    return sdg.split('\n')[0][7:]  # Removing '# ::id '


def remove_sdg_id(sdg):
    return '\n'.join(sdg.split('\n')[1:]) # Removing the ID line

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

    for i in range(len(data)):
        id = data[i]['id']
        if id not in sdg_dict:
            print("Sentence with ID='{}' has no SDG graph".format(id))
            continue
        
        data[i]['sdg'] = sdg_dict[id]

    with io.open(args.output_json, 'w', encoding='utf-8') as f:
        dumps = json.dumps(data, indent=True)
        f.write(dumps)

if __name__ == '__main__':
    main()
