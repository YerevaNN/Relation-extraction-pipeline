from __future__ import print_function
from __future__ import division

import io
import argparse
import json
from tqdm import tqdm
# from nltk.tokenize import wordpunct_tokenize
from nltk.metrics.distance import edit_distance

def word_id(sdg, word):
    if ' ' in word:
        # choose the first word
        word = word.split(' ')[0]
    sdg_words = []
    for line in sdg.split('\n'):
        id, sdg_word, _ , _ , _ , _ = parse_sdg_line(line)
        sdg_words.append((id, sdg_word))
    sdg_words = sorted(sdg_words, key=lambda x: edit_distance(x[1], word))
    # with open('tmplog', 'a', encoding='utf-8') as f:
    #     f.write(u'{} {} {}\n'.format(edit_distance(sdg_words[0][1], word), word, sdg_words[0][1]))
    return sdg_words[0][0]



def parse_sdg_line(sdg_line):
    line_arr = sdg_line.split('\t')
    id = int(line_arr[0])
    word = line_arr[1]
    lemma = line_arr[2]
    pos = line_arr[3]
    parent_id = int(line_arr[4])
    edge = line_arr[5]
    return id, word, lemma, pos, parent_id, edge


def word_in_path(path, word):
    for (i, (w, p, e)) in enumerate(path):
        if w == word:
            return i + 1
    return None


def sdg_line_by_id(sdg, id):
    for line in sdg.split('\n'):
        if int(line.split()[0]) == id:
            return line
    
    print("ID {} does not exist in \n {}".format(id, sdg))
    return None


def sdg_paths(sdg, word_1, word_2):
    id_1 = word_id(sdg, word_1)
    id_2 = word_id(sdg, word_2)
    path_1 = []
    path_2 = []
    sdg = sdg
    
    if id_1 is None or id_2 is None:
        return [], []
    
    id = id_1
    while id != 0:
        line = sdg_line_by_id(sdg, id)
        if line is None:
            return [], []
        _, word, _, pos, id, edge = parse_sdg_line(line)
        path_1.append((word, pos, edge))
    id = id_2
    while id != 0:
        line = sdg_line_by_id(sdg, id)
        if line is None:
            return [], []
        _, word, _, pos, id, edge = parse_sdg_line(line)
        path_2.append((word, pos, edge))
        index = word_in_path(path_1, word)
        if index:
            return path_1[ :index], path_2
    return [], []


def sentence_from_sdg_paths(paths):
    sentence = ''
    for (word, pos, edge) in paths[0][:-1]:
        sentence += ' '.join([word, edge]) + ' '

    if paths[0]:
        sentence += paths[0][-1][0] + ' '  # Not including edge from LCA to its parent
        sentence += '__LCA_TAG__ ' + paths[0][-1][0] + ' '  # Marking LCA
    elif paths[1]:
        sentence += '__LCA_TAG__ ' + paths[1][0][0] + ' '  # Marking LCA

    for (word, pos, edge) in paths[1][-2::-1]:  # Not including LCA at all
        sentence += ' '.join([edge, word]) + ' '
    return sentence[:-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_json', '-ij', required=True, type=str)
    parser.add_argument('--output_json', '-o', required=True, type=str)
    args = parser.parse_args()

    with io.open(args.input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for i in tqdm(range(len(data))):
        sdg = data[i]['sdg']
        if sdg.strip() == '':
            print("AAAA!")
            data[i]['extracted_information'][j]['sdg_path'] = ''
            continue
            
        for j in range(len(data[i]['extracted_information'])):
            info = data[i]['extracted_information'][j]
            if not info['participant_a']:
                word_1 = info['interaction_type']
                word_2 = info['participant_b']
            elif not info['participant_b']:
                word_1 = info['interaction_type']
                word_2 = info['participant_a']
            else:
                word_1 = info['participant_a']
                word_2 = info['participant_b']
            paths = sdg_paths(sdg, word_1, word_2)
            sentence = sentence_from_sdg_paths(paths)
            data[i]['extracted_information'][j]['sdg_path'] = sentence

    with io.open(args.output_json, 'w', encoding='utf-8') as f:
        data_string = json.dumps(data, indent=True)
        f.write(data_string)

if __name__ == '__main__':
    main()
