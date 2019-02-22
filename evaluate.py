from __future__ import print_function
from __future__ import division

import io
import json
import codecs
import argparse
import itertools
import re
from sentence_filters import multiword, tags
import soft_text_match as stm
import numpy as np

config = {
    "match_by": None
}


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
    if config['match_by'] == 'id':
        return item['id']
    elif config['match_by'] == 'line':
        return item['line']
    elif config['match_by'] == 'text':
        text = item['text']
        text = text.lower()
        text = re.sub('[^0-9a-z]+', '', text)
        return hash(text)
    else:
        raise Exception("Cannot compute hash for sentences")


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


def get_sentences(data):
    #    data = data.values()
    data = {
        hash_sentence(s): s for s in data
    }
    return data


def evaluate_sentences(truth_sentences, pred_sentences, keys=None):
    TP = 0
    FN = 0
    FP = 0

    if keys is None:
        keys = truth_sentences.keys()

    fp_entities = 0
    entity_version_mismatch = 0
    fp_interaction_due_to_entity = 0

    for id in keys:
        # match unique entities
        if id not in pred_sentences:
            print("No prediction for sentence with ID=".format(id))
            continue
        sp = pred_sentences[id]
        st = truth_sentences[id]

        pred_ue_to_truth_ue = {}

        for ue, ue_obj in sp['unique_entities']:
            for ve, ve_obj in ue_obj['versions'].items():
                if ve in st['entity_map']:
                    true_ue_id = st['entity_map'][ve]
                    if ue in pred_ue_to_truth_ue and pred_ue_to_truth_ue[ue] != true_ue_id:
                        # another version of this entity cluster was matched to a different cluster
                        entity_version_mismatch += 1
                    else:
                        pred_ue_to_truth_ue[ue] = true_ue_id
                else:
                    # this version does not exist in the ground truth
                    fp_entities += 1

        predicted_pairs = set()
        for interaction in sp['extracted_information']:
            pa, pb = interaction['participant_ids']
            if pa not in pred_ue_to_truth_ue:
                fp_interaction_due_to_entity += 1
            elif pb not in pred_ue_to_truth_ue:
                fp_interaction_due_to_entity += 1
            else:
                predicted_pairs.add(set([
                    pred_ue_to_truth_ue[pa],
                    pred_ue_to_truth_ue[pb],
                ]))

        truth_pairs = set([set(i['participant_ids']) for i in st['extracted_information']])

        common = truth_pairs.intersection(predicted_pairs)
        sentence_TP = len(common)
        sentence_FN = len(truth_pairs) - sentence_TP
        sentence_FP = len(predicted_pairs) - sentence_TP

        TP += sentence_TP
        FN += sentence_FN
        FP += sentence_FP

        # TODO: check labels!

    return TP, FN, FP, fp_entities, entity_version_mismatch, fp_interaction_due_to_entity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--truth_path', '-t', required=True, type=str)
    parser.add_argument('--prediction_path', '-p', nargs='*', required=True, type=str)

    parser.add_argument('--only', default='', type=str)
    parser.add_argument('--only_bind', action='store_true')
    parser.add_argument('--sentence_level', action='store_true')
    parser.add_argument('--include_negatives', action='store_true')
    parser.add_argument('--bootstrap_count', default=0, type=int)

    parser.add_argument('--multiword', default=0, type=int, help='values: +1 or -1')
    parser.add_argument('--tags', default='', type=str, help='example: complex+1,abbr-1')
    parser.add_argument('--has_sdg', default=0, type=int, help='+1 or -1')

    parser.add_argument('--sentence_stats', action='store_true')
    parser.add_argument('--match_by', '-mb', default='id', type=str,
                        choices=['id', 'text', 'line'])

    args = parser.parse_args()

    config['match_by'] = args.match_by

    print(args)

    positive_labels = [-1, 1] if args.include_negatives else [1]

    with io.open(args.truth_path, 'r', encoding='utf-8') as f:
        truth = json.load(f)

    predictions = []
    for p in args.prediction_path:
        print(p)
        with io.open(p, 'r', encoding='utf-8') as f:
            prediction = json.load(f)
            predictions.append(prediction)

    if args.only_bind:
        args.only = 'bind'

    if args.multiword != 0:
        raise Exception("Not implemented in `union` mode")
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
        raise Exception("Not implemented in `union` mode")
        truth = [t for t in truth if tags(t, args.tags)]
        prediction = [p for p in prediction if tags(p, args.tags)]

    if args.has_sdg != 0:
        raise Exception("Not implemented in `union` mode")
        if args.has_sdg == 1:
            has_sdg_filter = lambda x: x['sdg_path'] != ''
        if args.has_sdg == -1:
            has_sdg_filter = lambda x: x['sdg_path'] == ''

        for t in truth:
            t['extracted_information'] = [x for x in t['extracted_information'] if has_sdg_filter(x)]
        for p in prediction:
            p['extracted_information'] = [x for x in p['extracted_information'] if has_sdg_filter(x)]

    truth_sentences = get_sentences(truth)
    pred_sentences = get_sentences(prediction)
    print("Total true relations: {}".format(sum([len(ts) for ts in truth_sentences.values()])))

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

    TP, FN, FP, fp_entities, entity_version_mismatch, fp_interaction_due_to_entity = evaluate_sentences(
        truth_sentences, pred_sentences)

    print("\n")
    print("True Positive: {}".format(TP))
    print("False Negative: {}".format(FN))
    print("False Positive: {}".format(FP))
    print(" ")
    print("FP entities: {}".format(fp_entities))
    print("Entity version mismatch: {}".format(entity_version_mismatch))
    print("FP due to entities: {}".format(fp_interaction_due_to_entity))

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    fscore = 2 * precision * recall / (precision + recall)

    print("Precision: {:.2f}% \nRecall: {:.2f}% \nF-score: {:.2f}%".format(
        precision * 100, recall * 100, fscore * 100))


if __name__ == '__main__':
    main()
