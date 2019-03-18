#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division

import io
import json
import argparse
import re
import tqdm
import numpy as np
import sklearn.utils as sk_utils

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

def get_entity_coreferences(sentence):
    return {tuple(sorted([ve, ve2]))
                              for ue, ue_obj in sentence['unique_entities'].items()
                              for ve, ve_obj in ue_obj['versions'].items() if 'mentions' in ve_obj
                              for ve2 in ue_obj['versions'].keys() if ve2 != ve}

class PRFScores:
    def __init__(self, name):
        self.name = name
        self.TP = 0
        self.FN = 0
        self.FP = 0
        self.by_id = {}

    def store_by_id(self, id, TP, FN, FP):
        if id not in self.by_id:
            self.by_id[id] = PRFScores(self.name)
        self.by_id[id].TP += TP
        self.by_id[id].FN += FN
        self.by_id[id].FP += FP

    def add_sets(self, id, truth_set, prediction_set):
        common = truth_set.intersection(prediction_set)
        TP = len(common)
        FN = len(truth_set) - TP
        FP = len(prediction_set) - TP
        self.TP += TP
        self.FN += FN
        self.FP += FP
        self.store_by_id(id, TP, FN, FP)

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

        return {
            "precision": precision,
            "recall": recall,
            "fscore": fscore
        }

    def print_scores(self):
        print("\n{}".format(self.name))
        print("          | Pred 0 | Pred 1")
        print("   True 0 |        | {:>6}".format(self.FP))
        print("   True 1 | {:>6} | {:>6}".format(self.FN, self.TP))

        scores = self.return_scores()

        print("      Precision: {:>5.2f}%\n"
              "      Recall:    {:>5.2f}% \n"
              "      F-score:   {:>5.2f}%".format(
            scores['precision'] * 100, scores['recall'] * 100, scores['fscore'] * 100))


class PRFScoresFlatMentions(PRFScores):
    def add_sets(self, id, truth_set, prediction_set):
        common = truth_set.intersection(prediction_set)
        TP = len(common)
        # remove the ones which intersect with TPs
        T_intersects_with_TP = {(e,start,end) for e,start,end in truth_set
                               for c_e,c_start,c_end in common
                               if c_start < end and c_end > start and e != c_e}
        P_intersects_with_TP = {(e,start,end) for e,start,end in prediction_set
                               for c_e,c_start,c_end in common
                               if c_start < end and c_end > start and e != c_e}

        truth_set -= T_intersects_with_TP
        prediction_set -= P_intersects_with_TP
        # remove the ones that are in a larger entity
        T_contains_shorter = {(e1, start1, end1) for e1, start1, end1 in truth_set
                               for e2, start2, end2 in truth_set
                               if start1 <= start2 and end2 <= start1 and e1!=e2}
        P_contains_shorter = {(e1, start1, end1) for e1, start1, end1 in prediction_set
                               for e2, start2, end2 in prediction_set
                               if start1 <= start2 and end2 <= start1 and e1!=e2}

        truth_set -= T_contains_shorter
        prediction_set -= P_contains_shorter

        FN = len(truth_set) - TP
        FP = len(prediction_set) - TP
        self.TP += TP
        self.FN += FN
        self.FP += FP
        self.store_by_id(id, TP, FN, FP)


def evaluate_sentences(truth_sentences, pred_sentences, keys=None):
    relation_extraction_any_score = PRFScores('Relation Extraction (any)')
    relation_extraction_all_score = PRFScores('Relation Extraction (all)')
    entity_mentions_score = PRFScores('Entity Mentions')
    entity_mentions_flat_score = PRFScoresFlatMentions('Entity Mentions (flat)')
    entities_score = PRFScores("Entities")
    entity_coreferences_score = PRFScores("Entity Coreferences")

    if keys is None:
        keys = truth_sentences.keys()

    for id in keys:
        # match unique entities
        if id not in pred_sentences:
            print("No prediction for sentence with ID={}".format(id))
            continue
        sp = pred_sentences[id]
        st = truth_sentences[id]

        st_entity_mentions = get_entity_mentions(st)
        sp_entity_mentions = get_entity_mentions(sp)
        entity_mentions_score.add_sets(id, st_entity_mentions, sp_entity_mentions)
        entity_mentions_flat_score.add_sets(id, st_entity_mentions, sp_entity_mentions)

        st_entities = {e for e, start, end in st_entity_mentions}
        sp_entities = {e for e, start, end in sp_entity_mentions}
        entities_score.add_sets(id, st_entities, sp_entities)

        st_entity_coreferences = get_entity_coreferences(st)
        sp_entity_coreferences = get_entity_coreferences(sp)
        entity_coreferences_score.add_sets(id, st_entity_coreferences, sp_entity_coreferences)

        # pred_ue_to_truth_ue = {}
        #
        # for ue, ue_obj in sp['unique_entities'].items():
        #     ue = int(ue)
        #     for ve, ve_obj in ue_obj['versions'].items():
        #         if ve in st['entity_map']:
        #             true_ue_id = int(st['entity_map'][ve])
        #             if ue in pred_ue_to_truth_ue and pred_ue_to_truth_ue[ue] != true_ue_id:
        #                 # another version of this entity cluster was matched to a different cluster
        #                 entity_version_mismatch += 1
        #             else:
        #                 pred_ue_to_truth_ue[ue] = true_ue_id
        #         else:
        #             # pred_ue_to_truth_ue[ue] = -ue
        #             # this version does not exist in the ground truth
        #             fp_entities += 1

        # st_unique_entities = set([int(x) for x in st['unique_entities'].keys()])
        # sp_unique_entities = set(pred_ue_to_truth_ue.values())
        # unique_entities_score.add_sets(st_unique_entities, sp_unique_entities)

        # interactions
        predicted_pairs_with_names = {tuple(sorted([ve_a, ve_b]))
                for interaction in sp['extracted_information']
                for ve_a, ve_obj in sp['unique_entities'][str(interaction['participant_ids'][0])]['versions'].items()
                for ve_b, ve_obj in sp['unique_entities'][str(interaction['participant_ids'][1])]['versions'].items() }
        # sometimes duplicates exist

        predicted_pairs_with_names_matched = set()

        for interaction in st['extracted_information']:
            if interaction['contains_implicit_entity']:
                continue
            # if 'implicit' in interaction and interaction['implicit']:
            #     continue
            ta, tb = interaction['participant_ids']
            true_pairs_with_names = {tuple(sorted([ve_a, ve_b]))
                for ve_a, ve_aobj in st['unique_entities'][str(ta)]['versions'].items()
                                      if ve_aobj['exists'] == True
                for ve_b, ve_bobj in st['unique_entities'][str(tb)]['versions'].items()
                                      if ve_bobj['exists'] == True
                                     } # no duplicates detected

            intersect = true_pairs_with_names.intersection(predicted_pairs_with_names)
            predicted_pairs_with_names_matched = predicted_pairs_with_names_matched.union(intersect)

            true_to_add = {tuple(sorted([ta, tb]))}
            predicted_any_to_add = set()
            predicted_all_to_add = set()

            if len(intersect) > 0:
                predicted_any_to_add = true_to_add

            if len(intersect) == len(true_pairs_with_names):
                predicted_all_to_add = true_to_add

            relation_extraction_any_score.add_sets(id, true_to_add, predicted_any_to_add)
            relation_extraction_all_score.add_sets(id, true_to_add, predicted_all_to_add)

        predicted_pairs_with_names_unmatched = predicted_pairs_with_names - predicted_pairs_with_names_matched
        relation_extraction_any_score.add_sets(id, set(), predicted_pairs_with_names_unmatched)
        relation_extraction_all_score.add_sets(id, set(), predicted_pairs_with_names_unmatched)

        # TODO: check labels!

    return relation_extraction_any_score, relation_extraction_all_score, entity_mentions_score, entity_mentions_flat_score, entities_score, entity_coreferences_score


class BootstrapEvaluation:
    def __init__(self, truth_objects, prediction_objects, evaluate_fn, bootstrap_count):
        self.bootstrap_count = bootstrap_count
        self.truth = truth_objects
        self.prediction_dict = prediction_objects
        self.evaluate_fn = evaluate_fn
        self.runs = {}
        self.results = {}
        self.score_types = ['precision', 'recall', 'fscore']

    def initialize_runs(self, name):
        self.runs[name] = {filename: {
            score_type: []
            for score_type in self.score_types
        } for filename in self.prediction_dict.keys()}

    def add_run(self, filename, score):
        scores = score.return_scores()
        for score_type in self.score_types:
            self.runs[score.name][filename][score_type].append(scores[score_type])

    def evaluate(self):
        keys = list(self.truth.keys())
        print("Starting to bootstrap for {} times".format(self.bootstrap_count))
        for i in tqdm.tqdm(range(self.bootstrap_count)):
            cur_keys = sk_utils.resample(keys, n_samples=len(keys))
            for filename, prediction in self.prediction_dict.items():
                all_scores = self.evaluate_fn(self.truth, prediction, cur_keys)
                for score in all_scores:
                    if not isinstance(score, PRFScores):
                        continue
                    if score.name not in self.runs:
                        self.initialize_runs(score.name)
                    self.add_run(filename, score)

        self.results = {}
        for score_name, score_data in self.runs.items():
            self.results[score_name] = {}
            for filename in self.prediction_dict.keys():
                self.results[score_name][filename] = {}
                for score_type, values in score_data[filename].items():
                    self.results[score_name][filename][score_type] = {
                        'mean': np.mean(values),
                        'median': np.median(values),
                        'std': np.std(values),
                        '2.5%': np.percentile(values, 2.5),
                        '97.5%': np.percentile(values, 97.5),
                    }

        print("Bootstrapping completed")

        return self.results

    def print_results(self):
        for filename in self.prediction_dict.keys():
            print("\n{}".format(filename))
            for score_name, score_obj in self.results.items():
                print("   {} (n={})".format(score_name, self.bootstrap_count))
                for score_type, score_stats in score_obj[filename].items():
                    print(u"      {:<10} {:>5.2f} ± {:>5.2f} ({:5.2f} - {:5.2f})".format(
                        score_type,
                        100 * score_stats['mean'],
                        100 * score_stats['std'],
                        100 * score_stats['2.5%'],
                        100 * score_stats['97.5%'],
                    ))

        for score_name, score_obj in self.results.items():
            print("\n{} (n={})".format(score_name, self.bootstrap_count))
            for score_type in self.score_types:
                print("   {:<10} {:>23}: ".format(score_type, ' '), end='')
                for i in range(len(self.prediction_dict)):
                    print("({}) ".format(i+1), end='')
                print(" ")
                for i1, filename1 in enumerate(self.prediction_dict.keys()):
                    print("   ({}) {:>30}: ".format(i1+1, filename1[-30:]), end='')
                    for filename2 in self.prediction_dict.keys():
                        if filename1 == filename2:
                            cell = ''
                        else:
                            cell = len([1 for i in range(self.bootstrap_count) if self.runs[score_name][filename1][score_type][i] >= self.runs[score_name][filename2][score_type][i]])
                        print("{:>3} ".format(cell), end='')
                    print(" ")

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

    predictions = {}
    for p in args.prediction_path:
        with io.open(p, 'r', encoding='utf-8') as f:
            prediction = json.load(f)
            predictions[p] = prediction

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
    print("{} truth sentences read from {}. {} objects extracted".format(len(truth), args.truth_path, len(truth_sentences)))
    pred_sentences_dict = {}
    for filename, prediction in predictions.items():
        pred_sentences = get_sentences(prediction)
        print("{} pred sentences read from {}. {} objects extracted".format(len(prediction), filename, len(pred_sentences)))
        pred_sentences_dict[filename] = pred_sentences

    if args.bootstrap_count > 0:
        be = BootstrapEvaluation(truth_sentences, pred_sentences_dict, evaluate_sentences, args.bootstrap_count)
        results = be.evaluate()
        be.print_results()

    for filename, pred_sentences in pred_sentences_dict.items():
        print("\n" + "=" * 80)
        print("Results for {}:".format(filename))
        scores = evaluate_sentences(truth_sentences, pred_sentences)

        for score in scores:
            score.print_scores()

        sentences_with_scores = []
        for sentence in pred_sentences.values():
            sentence['scores'] = {}
            for score in scores:
                sentence['scores'][score.name] = score.by_id[hash_sentence(sentence)].return_scores()
            sentences_with_scores.append(sentence)

        with open(filename + "_scores", 'w') as f:
            json.dump(sentences_with_scores, f, indent=True)

if __name__ == '__main__':
    main()
