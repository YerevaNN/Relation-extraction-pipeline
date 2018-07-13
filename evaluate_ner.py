from __future__ import print_function
from __future__ import division

import json
import codecs
import argparse
import itertools
import re
from sentence_filters import multiword

def normalize(text, bool_normalize):
    if not bool_normalize:
        return text
    text = text.lower()
    tokens = re.split('([-.,;\(\)\s])', text)
    tokens = [s for s in tokens if s.strip() not in ['', ',', '.', '(', ')']]
    return ' '.join(tokens)

def hash_sentence(item):
    text = item['text']
    text = text.lower()
    text = re.sub('[^0-9a-z]+', '', text)
    return hash(text)

def get_all_entities(data):
    return [
        [normalize(e) for e in v['entities']]
        for v in data        
    ]

def get_entities(data, bool_normalize):
    return [
        normalize(e, bool_normalize)
        for sentence in data
        for e in sentence['entities']
    ]

def get_sentences(data, normalize):
#    data = data.values()
    data = sorted(data, key=hash_sentence)
    grouped_data = itertools.groupby(data, key=hash_sentence)
    grouped_data = {key: frozenset(get_entities(data, normalize))
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
        sentence_truth_entities = truth_sentences.get(sentence)
        sentence_pred_entities = pred_sentences.get(sentence, frozenset())
        
        common = sentence_truth_entities.intersection(sentence_pred_entities)
        sentence_TP = len(common)
        sentence_FN = len(sentence_truth_entities) - sentence_TP
        sentence_FP = len(sentence_pred_entities) - sentence_TP

        TP += sentence_TP
        FN += sentence_FN
        FP += sentence_FP
    
    return TP, FN, FP

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--truth_path', '-t', required=True, type=str)
    parser.add_argument('--prediction_path', '-p', required=True, type=str)
    parser.add_argument('--sentence_level', action='store_true')
    parser.add_argument('--bootstrap_count', default=0, type=int)
    parser.add_argument('--normalize', '-n', action='store_true')
    
    parser.add_argument('--multiword', default=0, type=int, help='values: +1 or -1')
    args = parser.parse_args()

    with codecs.open(args.truth_path, 'r', encoding='utf-8') as f:
        truth = json.load(f)

    with codecs.open(args.prediction_path, 'r', encoding='utf-8') as f:
        predictions = json.load(f)
        
    if args.multiword != 0:
        truth = [t for t in truth if multiword(t, args.multiword)]
        predictions = [p for p in predictions if multiword(p, args.multiword)]
        
    if args.sentence_level:
        truth_sentences = get_sentences(truth, args.normalize)
        pred_sentences = get_sentences(predictions, args.normalize)
        print("{} truth sentences read from json. {} objects extracted".format(len(truth), len(truth_sentences)))
        print("{} pred sentences read from json. {} objects extracted".format(len(predictions), len(pred_sentences)))
        
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

    else:
        truth_entities = get_all_entities(truth, args.normalize)
        predicted_entities = get_all_entities(predictions, args.normalize)

        assert(truth_entities)
        assert(predicted_entities)

        truth_entities_set = set().union(*truth_entities)
        predicted_entities_set = set().union(*predicted_entities)

        print("{} unique truth entities found".format(len(truth_entities_set)))
        print("{} unique predicted entities found".format(len(predicted_entities_set)))

        intersection = set.intersection(truth_entities_set, predicted_entities_set)
        union = set.union(truth_entities_set, predicted_entities_set)

        TP = len(intersection)
        FN = len(truth_entities_set) - TP
        FP = len(predicted_entities_set) - TP

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
