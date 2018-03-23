from __future__ import print_function
from __future__ import division

import io
import argparse
import json
import os
from subprocess import check_call

def amr_id(amr):
    return amr.split('\n')[0][7:]  # Removing '# ::id '

def amr_body(amr):
    return '\n'.join(amr.split('\n')[2:])  # Removing id and snt lines

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_text', '-it', required=True, type=str)
    parser.add_argument('--input_json', '-ij', required=True, type=str)
    parser.add_argument('--model', '-m', required=True, type=str)
    parser.add_argument('--tmp_dir', '-t', required=True, type=str)
    parser.add_argument('--output_json', '-o', required=True, type=str)
    args = parser.parse_args()

    input_text_abs_path = os.path.abspath(args.input_text)
    parsed_amr_path = os.path.join(args.tmp_dir, 'amrs')
    parsed_amr_path = os.path.abspath(parsed_amr_path)
    check_call(['docker', 'run', '-it', '--rm',
                '-v', input_text_abs_path + ':/tmp/in/input.txt:ro',
                '-v', parsed_amr_path + ':/tmp/out:rw',
                'yerevann/camr', args.model])

    parsed_amr_path = os.path.join(parsed_amr_path, 'output.txt')
    with io.open(parsed_amr_path, 'r', encoding='utf-8') as f:
        amrs = f.read().split('\n\n')[:-1]  # Removing last empty sentence

    amr_dict = {amr_id(amr) : amr_body(amr) for amr in amrs}

    with io.open(args.input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for i in range(len(data)):
        id = data[i]['id']
        data[i]['amr'] = amr_dict[id]

    with io.open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=True)

if __name__ == '__main__':
    main()
