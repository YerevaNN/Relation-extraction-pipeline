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
    args = parser.parse_args()

    sdg_output = os.path.join(args.tmp_dir, 'output.conll')
    check_call(['bash', 'submodules/conll_parser/parse_script.sh',
                args.input_text, sdg_output])

    with io.open(sdg_output, 'r', encoding='utf-8') as f:
        sdgs = f.read().split('\n\n')
       
    if sdgs[-1].strip():
        sdgs = sdgs[:-1]

    sdg_dict = {sdg_id(sdg) : remove_sdg_id(sdg) for sdg in sdgs}

    with io.open(args.input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for i in range(len(data)):
        id = data[i]['id']
        data[i]['sdg'] = sdg_dict[id]

    with io.open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=True)

if __name__ == '__main__':
    main()
