#!/usr/bin/python
# -*- coding: utf-8 -*-

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
    if config['match_by'] == 'text':
        text = item['text']
        text = text.lower()
        text = re.sub('[^0-9a-z]+', '', text)
        return hash(text)
    else:
        return str(item[config['match_by']])


def get_sentences(data):
    #    data = data.values()
    data = {
        hash_sentence(s): s for s in data
    }
    return data


def get_entity_mentions(sentence):
    return {(ve, mention[0], mention[1])
                              for ue, ue_obj in sentence['unique_entities'].items()
                              for ve, ve_obj in ue_obj['versions'].items() if 'mentions' in ve_obj
                              for mention in ve_obj['mentions']}

class PRFScores:
    def __init__(self, name):
        self.name = name
        self.TP = 0
        self.FN = 0
        self.FP = 0

    def add_sets(self, truth_set, prediction_set):
        common = truth_set.intersection(prediction_set)
        TP = len(common)
        FN = len(truth_set) - TP
        FP = len(prediction_set) - TP
        self.TP += TP
        self.FN += FN
        self.FP += FP

    def return_scores(self):
        if self.TP + self.FP == 0:
            precision = 0
        else:
            precision = self.TP / (self.TP + self.FP)
        if self.TP + self.FN == 0:
            recall = 0
        else:
            recall = self.TP / (self.TP + self.FN)
        if precision + recall == 0:
            fscore = 0
        else:
            fscore = 2 * precision * recall / (precision + recall)

        return precision, recall, fscore

    def print_scores(self):
        print("\n{}".format(self.name))
        print("          | Pred 0 | Pred 1")
        print("   True 0 |        | {:>6}".format(self.FP))
        print("   True 1 | {:>6} | {:>6}".format(self.FN, self.TP))

        precision, recall, fscore = self.return_scores()

        print("   Precision: {:.2f}% \n   Recall: {:.2f}% \n   F-score: {:.2f}%".format(
            precision * 100, recall * 100, fscore * 100))


def evaluate_sentences(truth_sentences, pred_sentences, keys=None):
    relation_extraction_score = PRFScores('Relation extraction')
    entity_mentions_score = PRFScores('Entity mentions')
    entities_score = PRFScores("Entities")

    if keys is None:
        keys = truth_sentences.keys()

    fp_entities = 0
    entity_version_mismatch = 0
    fp_interaction_due_to_entity = 0

    for id in keys:
        # match unique entities
        if id not in pred_sentences:
            print("No prediction for sentence with ID={}".format(id))
            continue
        sp = pred_sentences[id]
        st = truth_sentences[id]

        sp_entity_mentions = get_entity_mentions(sp)
        st_entity_mentions = get_entity_mentions(st)
        entity_mentions_score.add_sets(st_entity_mentions, sp_entity_mentions)

        sp_entities = {e for e, start, end in sp_entity_mentions}
        st_entities = {e for e, start, end in st_entity_mentions}
        entities_score.add_sets(st_entities, sp_entities)

        pred_ue_to_truth_ue = {}

        for ue, ue_obj in sp['unique_entities'].items():
            ue = int(ue)
            for ve, ve_obj in ue_obj['versions'].items():
                if ve in st['entity_map']:
                    true_ue_id = int(st['entity_map'][ve])
                    if ue in pred_ue_to_truth_ue and pred_ue_to_truth_ue[ue] != true_ue_id:
                        # another version of this entity cluster was matched to a different cluster
                        entity_version_mismatch += 1
                    else:
                        pred_ue_to_truth_ue[ue] = true_ue_id
                else:
                    # this version does not exist in the ground truth
                    fp_entities += 1

        # interactions
        truth_pairs = set([tuple(sorted(i['participant_ids'])) for i in st['extracted_information']])
        predicted_pairs = set()
        for interaction in sp['extracted_information']:
            pa, pb = interaction['participant_ids']
            if pa not in pred_ue_to_truth_ue:
                fp_interaction_due_to_entity += 1
                ta = -1
                tb = -1
            elif pb not in pred_ue_to_truth_ue:
                fp_interaction_due_to_entity += 1
                ta = -1
                tb = -1
            else:
                ta = pred_ue_to_truth_ue[pa]
                tb = pred_ue_to_truth_ue[pb]
            predicted_pairs.add(tuple(sorted([ta, tb])))

        relation_extraction_score.add_sets(truth_pairs, predicted_pairs)

        # TODO: check labels!

    return relation_extraction_score, entity_mentions_score, entities_score, fp_entities, entity_version_mismatch, fp_interaction_due_to_entity


import sklearn.utils as sk_utils
class BootstrapEvaluation:
    def __init__(self, truth_objects, prediction_objects, evaluate_fn, bootstrap_count):
        self.bootstrap_count = bootstrap_count
        self.truth = truth_objects
        self.prediction = prediction_objects
        self.evaluate_fn = evaluate_fn
        self.runs = {}

    def initialize_runs(self, name):
        self.runs[name] = {
            "precision": [],
            "recall": [],
            "fscore": []
        }

    def add_run(self, score):
        precision, recall, fscore = score.return_scores()
        self.runs[score.name]['precision'].append(precision)
        self.runs[score.name]['recall'].append(recall)
        self.runs[score.name]['fscore'].append(fscore)

    def evaluate(self):
        keys = list(self.truth.keys())
        print("Starting to bootstrap for {} times".format(self.bootstrap_count))
        for i in range(self.bootstrap_count):
            cur_keys = sk_utils.resample(keys, n_samples=len(keys))
            all_scores = self.evaluate_fn(self.truth, self.prediction, cur_keys)
            for score in all_scores:
                if not isinstance(score, PRFScores):
                    continue
                if score.name not in self.runs:
                    self.initialize_runs(score.name)
                self.add_run(score)

        self.results = {}
        for score_name, score_data in self.runs.items():
            self.results[score_name] = {}
            for score_type, values in score_data.items():
                self.results[score_name][score_type] = {
                    'mean': np.mean(values),
                    'median': np.median(values),
                    'std': np.std(values),
                    '2.5%': np.percentile(values, 2.5),
                    '97.5%': np.percentile(values, 97.5),
                }

        print("Bootstrapping completed")

        return self.results

    def print_results(self):
        for score_name, score_obj in self.results.items():
            print("\n{}".format(score_name))
            for score_type, score_stats in score_obj.items():
                print(u"   {:<10} {:.2f} ± {:.2f} ({:.2f} - {:.2f})".format(
                    score_type,
                    100 * score_stats['mean'],
                    100 * score_stats['std'],
                    100 * score_stats['2.5%'],
                    100 * score_stats['97.5%'],
                ))


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
    parser.add_argument('--match_by', '-mb', default='id', type=str)

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
        be = BootstrapEvaluation(truth_sentences, pred_sentences, evaluate_sentences, args.bootstrap_count)
        results = be.evaluate()
        be.print_results()

    relation_extraction_score, entity_mentions_score, entities_score, fp_entities, entity_version_mismatch, fp_interaction_due_to_entity = evaluate_sentences(
        truth_sentences, pred_sentences)

    print(" ")
    print("FP entities: {}".format(fp_entities))
    print("Entity version mismatch: {}".format(entity_version_mismatch))
    print("FP due to entities: {}".format(fp_interaction_due_to_entity))

    relation_extraction_score.print_scores()
    entity_mentions_score.print_scores()
    entities_score.print_scores()


if __name__ == '__main__':
    main()
