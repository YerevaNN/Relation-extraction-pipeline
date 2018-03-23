from __future__ import print_function
from __future__ import division

import io
import argparse
import json
import os
from subprocess import check_call


def paths_by_id(extracted_info, id):
    paths = []
    for info in extracted_info:
        if info[0] == id:
            paths.append(info[1])
    return paths


def sentence_from_path(path):
    sentence = ''

    if path[0]:
        sentence = path[0][0][0]
        for i in path[0][1:]:
            sentence += ' ' + i[1] + ' ' + i[0]

    if path[1]:
        for i in path[1][:1:-1]:
            sentence += ' ' + i[0] + ' ' + i[1]
        sentence += ' ' + path[1][0][0]

    return sentence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_json', '-ij', required=True, type=str)
    parser.add_argument('--output_json', '-o', required=True, type=str)
    parser.add_argument('--tmp_dir', '-t', required=True, type=str)
    args = parser.parse_args()

    with io.open(args.input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    extractor_input = []
    for sentence in data:
        amr = sentence['amr']
        id = sentence['id']
        index = 0
        for info in sentence['extracted_information']:
            if not info['participant_a']:
                extractor_input.append([id + '_' + str(index), amr,
                                        [info['interaction_type'],
                                         info['participant_b']]])
            elif not info['participant_b']:
                extractor_input.append([id + '_' + str(index), amr,
                                        [info['interaction_type'],
                                         info['participant_a']]])
            else:
                extractor_input.append([id + '_' + str(index), amr,
                                        [info['participant_a'],
                                         info['participant_b']]])
            index += 1

    paths_in_fname = os.path.join(args.tmp_dir, 'paths_in.json')
    paths_out_fname = os.path.join(args.tmp_dir, 'paths_out.json')

    with io.open(paths_in_fname, 'w', encoding='utf-8') as f:
        json.dump(extractor_input, f)

    check_call(['python3', 'extract_amr_paths.py',
                '-i', paths_in_fname,
                '-o', paths_out_fname])

    with io.open(paths_out_fname, 'r', encoding='utf-8') as f:
        extracted_info = json.load(f)

    for i in range(len(data)):
        id = data[i]['id']
        for index in range(len(data[i]['extracted_information'])):
            path = paths_by_id(extracted_info, id + '_' + str(index))[0]  # Just taking the first found path
            sent = sentence_from_path(path)
            data[i]['extracted_information'][index]['amr_path'] = sent

    with io.open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=True)

if __name__ == '__main__':
    main()
