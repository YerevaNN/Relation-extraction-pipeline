from __future__ import print_function
from __future__ import division

import json
import codecs
import argparse
import itertools
import re
from sentence_filters import multiword, tags

def normalize(text):
    if text:
        text = text.lower()
        tokens = re.split('([-.,;\(\)\s\?\!\/\*])', text)
        tokens = [s for s in tokens if s.strip() not in ['', ',', '.', '(', ')', '*', '/', '?', '!']]
        return ' '.join(tokens)
    return text

def normalize_set(interaction_tuple):
    interaction_tuple = [normalize(t) for t in interaction_tuple]
    return frozenset(interaction_tuple)

def hash_sentence(item):
    text = item['text']
    text = text.lower()
    text = re.sub('[^0-9a-z]+', '', text)
    return hash(text)

def get_true_tuples(data, positive_labels):
    return [
        (v['interaction_type'], normalize_set([v['participant_a'], v['participant_b']]))
        for sentence in data
        for v in sentence['extracted_information']
        if v['label'] in positive_labels
    ]

def get_bind_tuples(data):
    return [
        interaction_tuple for interaction_type, interaction_tuple
        in get_true_tuples(data)
        if interaction_type.startswith('bind')
    ]

def get_all_tuples(data, positive_labels, only=''):
    return [
        interaction_tuple
        for interaction_type, interaction_tuple
        in get_true_tuples(data, positive_labels)
        if interaction_type.startswith(only)
    ]

def get_sentences(data, positive_labels, only=''):
#    data = data.values()
    data = sorted(data, key=hash_sentence)
    grouped_data = itertools.groupby(data, key=hash_sentence)
    grouped_data = {key: frozenset(get_all_tuples(data, positive_labels, only=only))
                    for key, data
                    in grouped_data}
    return grouped_data


def evaluate_sentences(truth_sentences, pred_sentences, keys=None):
    TP = 0
    FN = 0
    FP = 0
    
    if keys is None:
        keys = truth_sentences.keys()

    for sentence in keys:
        sentence_truth_tuples = truth_sentences.get(sentence)
        sentence_pred_tuples = pred_sentences.get(sentence, frozenset())

        common = sentence_truth_tuples.intersection(sentence_pred_tuples)
        sentence_TP = len(common)
        sentence_FN = len(sentence_truth_tuples) - sentence_TP
        sentence_FP = len(sentence_pred_tuples) - sentence_TP

        TP += sentence_TP
        FN += sentence_FN
        FP += sentence_FP
    
    return TP, FN, FP


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--truth_path', '-t', required=True, type=str)
    parser.add_argument('--prediction_path', '-p', required=True, type=str)
    parser.add_argument('--only', default='', type=str)
    parser.add_argument('--only_bind', action='store_true')
    parser.add_argument('--sentence_level', action='store_true')
    parser.add_argument('--include_negatives', action='store_true')
    parser.add_argument('--bootstrap_count', default=0, type=int)
    
    parser.add_argument('--multiword', default=0, type=int, help='values: +1 or -1')
    parser.add_argument('--tags', default='', type=str, help='example: complex+1,abbr-1')
    parser.add_argument('--has_sdg', default=0, type=int, help='+1 or -1')
    
    args = parser.parse_args()
    
    print(args)
    
    positive_labels = [-1, 1] if args.include_negatives else [1]

    with codecs.open(args.truth_path, 'r', encoding='utf-8') as f:
        truth = json.load(f)

    with codecs.open(args.prediction_path, 'r', encoding='utf-8') as f:
        prediction = json.load(f)

    if args.only_bind:
        args.only = 'bind'

        
    if args.multiword != 0:
        # print("Multiword: {}".format(args.multiword))
        filtered_truth = []
        filtered_prediction = []
        for t in truth:
            if multiword(t, args.multiword):
                filtered_truth.append(t)
                h = hash_sentence(t)
                for p in prediction:
                    if hash_sentence(p) == h:
                        filtered_prediction.append(p)

        truth = filtered_truth
        prediction = filtered_prediction

    if args.tags:
        truth = [t for t in truth if tags(t, args.tags)]
        prediction = [p for p in prediction if tags(p, args.tags)]
       
    if args.has_sdg != 0:
        if args.has_sdg == 1:
            has_sdg_filter = lambda x: x['sdg_path'] != ''
        if args.has_sdg == -1:
            has_sdg_filter = lambda x: x['sdg_path'] == ''
            
        for t in truth:
            t['extracted_information'] = [x for x in t['extracted_information'] if has_sdg_filter(x)]
        for p in prediction:
            p['extracted_information'] = [x for x in p['extracted_information'] if has_sdg_filter(x)]

    if not args.sentence_level:
        truth_tuples = get_all_tuples(truth, positive_labels, only=args.only)
        prediction_tuples = get_all_tuples(prediction, positive_labels, only=args.only)

        print("{} truth tuples found".format(len(truth_tuples)))
        print("{} prediction tuples found".format(len(prediction_tuples)))

        truth_tuples_set = set(truth_tuples)
        prediction_tuples_set = set(prediction_tuples)

        print("{} unique truth tuples found".format(len(truth_tuples_set)))
        print("{} unique prediction tuples found".format(len(prediction_tuples_set)))
        
        if args.bootstrap_count > 0:
            print("Bootstrapping is not implemented for this setup")

        intersection = truth_tuples_set.intersection(prediction_tuples_set)
        union = truth_tuples_set.union(prediction_tuples_set)

        TP = len(intersection)
        FN = len(truth_tuples_set) - TP
        FP = len(prediction_tuples_set) - TP
        
        

    else:
#        raise "Not implemented"
        truth_sentences = get_sentences(truth, positive_labels, only=args.only)
        pred_sentences = get_sentences(prediction, positive_labels, only=args.only)
        print("{} truth sentences read from json. {} objects extracted".format(len(truth), len(truth_sentences)))
        print("{} pred sentences read from json. {} objects extracted".format(len(prediction), len(pred_sentences)))
        
        if args.bootstrap_count > 0:
            import numpy as np
            import sklearn.utils as sk_utils
            results = {
                "precision": {"runs": []},
                "recall": {"runs": []},
                "fscore": {"runs": []}
            }
            keys = list(truth_sentences.keys())
            print("Starting to bootstrap for {} times".format(args.bootstrap_count))
            for i in range(args.bootstrap_count):
                cur_keys = sk_utils.resample(keys, n_samples=len(keys))
                TP, FN, FP = evaluate_sentences(truth_sentences, pred_sentences, cur_keys)
                precision = TP / (TP + FP)
                recall = TP / (TP + FN)
                fscore = 2 * precision * recall / (precision + recall)
                results["precision"]['runs'].append(precision)
                results["recall"]['runs'].append(recall)
                results["fscore"]['runs'].append(fscore)

            for (m, values) in results.items():
                runs = results[m]['runs']
                results[m]['mean'] = np.mean(runs)
                results[m]['median'] = np.median(runs)
                results[m]['std'] = np.std(runs)
                results[m]['2.5% percentile'] = np.percentile(runs, 2.5)
                results[m]['97.5% percentile'] = np.percentile(runs, 97.5)
                del results[m]['runs']
            
            print(json.dumps(results, indent=True))
            
        print("Bootstrapping completed")

        TP, FN, FP = evaluate_sentences(truth_sentences, pred_sentences)

    print("\n")
    print("True Positive: {}".format(TP))
    print("False Negative: {}".format(FN))
    print("False Positive: {}".format(FP))

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    fscore = 2 * precision * recall / (precision + recall)

    print("Precision: {:.2f}% \nRecall: {:.2f}% \nF-score: {:.2f}%".format(
        precision*100, recall*100, fscore*100))


if __name__ == '__main__':
    main()
