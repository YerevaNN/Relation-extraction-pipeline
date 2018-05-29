from __future__ import print_function
from __future__ import division

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

def get_sentences(data, positive_labels, only=''):
#    data = data.values()
    data = sorted(data, key=hash_sentence)
    grouped_data = itertools.groupby(data, key=hash_sentence)
    grouped_data = {key: frozenset(get_all_tuples(data, positive_labels, only=only))
                    for key, data
                    in grouped_data}
    return grouped_data


def evaluate_sentences(truth_sentences, pred_sentences, keys=None, is_soft_match=False):
    TP = 0
    FN = 0
    FP = 0
    
    if keys is None:
        keys = truth_sentences.keys()

    for sentence in keys:
        sentence_truth_tuples = truth_sentences.get(sentence)
        sentence_pred_tuples = pred_sentences.get(sentence, frozenset())

        if not is_soft_match:
            common = sentence_truth_tuples.intersection(sentence_pred_tuples)
            sentence_TP = len(common)
            sentence_FN = len(sentence_truth_tuples) - sentence_TP
            sentence_FP = len(sentence_pred_tuples) - sentence_TP
        else:
            sentence_TP, sentence_FN, sentence_FP = evaluate_soft_match(
                gold_tuples=list(sentence_truth_tuples),
                prediction_tuples=list(sentence_pred_tuples),
            )

        TP += sentence_TP
        FN += sentence_FN
        FP += sentence_FP
    
    return TP, FN, FP             


def soft_match_wrapper(term, gold_term, stm_obj):
    #
    if (term is None) or (gold_term is None):
        assert not ((term is None) and (gold_term is None))
        return None
    else:
        return stm_obj.find_max_match_term(term, [gold_term])


def soft_match_tuple(
        tuple,
        gold_tuple,
        stm_obj_entity,
):
    #
    # print tuple
    # print gold_tuple
    assert len(tuple) == 2, tuple
    assert len(gold_tuple) == 2, gold_tuple
    #
    match_score = 0.0
    #
    assert tuple[0] is not None
    assert gold_tuple[0] is not None
    matched_protein = soft_match_wrapper(tuple[0], gold_tuple[0], stm_obj_entity)
    if matched_protein is None:
        return 0.0
    else:
        match_score += matched_protein[2]
    #
    assert tuple[1] is not None
    assert gold_tuple[1] is not None
    matched_protein2 = soft_match_wrapper(tuple[1], gold_tuple[1], stm_obj_entity)
    if matched_protein2 is None:
        return 0.0
    else:
        match_score += matched_protein2[2]
    #
    return match_score/len(tuple)


def evaluate_soft_match(gold_tuples, prediction_tuples):
    # 
    gold_tuples = [curr_gold_tuple for curr_gold_tuple in gold_tuples if len(curr_gold_tuple) == 2]
    prediction_tuples = [curr_tuple for curr_tuple in prediction_tuples if len(curr_tuple) == 2]
    # 
    len_gold_tuples = len(gold_tuples)
    len_prediction_tuples = len(prediction_tuples)
    #
    if len_gold_tuples == 0:
        return 0, 0, len_prediction_tuples
    # 
    stm_obj_entity = stm.SoftTextMatch(min_match_ratio=0.7, is_substring_match=True, is_stem=False)
    match_gold_idx = []
    #
    tp = 0
    #
    for curr_tuple in prediction_tuples:
        #
        curr_tuple = list(curr_tuple)
        #
        match_scores = np.zeros(len_gold_tuples)
        for curr_gold_idx in range(len_gold_tuples):
            curr_gold_tuple = gold_tuples[curr_gold_idx]
            curr_gold_tuple = list(curr_gold_tuple)
            curr_match_score = soft_match_tuple(curr_tuple, curr_gold_tuple, stm_obj_entity)
            curr_match_score_rev = soft_match_tuple(curr_tuple, list(reversed(curr_gold_tuple)), stm_obj_entity)
            match_scores[curr_gold_idx] = max(curr_match_score, curr_match_score_rev)
            curr_match_score = None
            curr_match_score_rev = None
        #
        max_match_score = match_scores.max()
        if max_match_score > 0.8:
            tp += 1
            curr_match_gold_idx = match_scores.argmax()

            # print gold_tuples[curr_match_gold_idx]

            if curr_match_gold_idx not in match_gold_idx:
                match_gold_idx.append(curr_match_gold_idx)
    #
    fn = len_gold_tuples - tp
    fp = len_prediction_tuples - tp
    #
    return tp, fn, fp


def main():
    print("\nTODO: Merge this script with evaluate.py\n")
    parser = argparse.ArgumentParser()
    parser.add_argument('--truth_path', '-t', required=True, type=str)
    parser.add_argument('--prediction_path', '-p', nargs='+', required=True, type=str)
    
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
                        choices=['id', 'text'])
                        
    parser.add_argument('--soft_match', '-sm', action='store_true', default=False)

    args = parser.parse_args()
    
    config['match_by'] = args.match_by
    
    print(args)
    
    positive_labels = [-1, 1] if args.include_negatives else [1]

    with codecs.open(args.truth_path, 'r', encoding='utf-8') as f:
        truth = json.load(f)

    predictions = []
    for p in args.prediction_path:
        with codecs.open(p, 'r', encoding='utf-8') as f:
            prediction = json.load(f)
            predictions.append(prediction)

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

        if not args.soft_match:
            intersection = truth_tuples_set.intersection(prediction_tuples_set)
            union = truth_tuples_set.union(prediction_tuples_set)

            TP = len(intersection)
            FN = len(truth_tuples_set) - TP
            FP = len(prediction_tuples_set) - TP
        else:
            TP, FN, FP = evaluate_soft_match(gold_tuples=truth_tuples_set, prediction_tuples=prediction_tuples_set)
    else:
#        raise "Not implemented"
        truth_sentences = get_sentences(truth, positive_labels, only=args.only)
        pred_sentences = {}
        for prediction in predictions:
            pred_sentences_current = get_sentences(prediction, positive_labels, only=args.only)
            for key, pairs in pred_sentences_current.items():
                if key not in pred_sentences:
                    pred_sentences[key] = set()
                pred_sentences[key] = pred_sentences[key].union(pairs)  #set([x for x in pairs])
    
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
                TP, FN, FP = evaluate_sentences(truth_sentences, pred_sentences, cur_keys, is_soft_match=args.soft_match)
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

        TP, FN, FP = evaluate_sentences(truth_sentences, pred_sentences, is_soft_match=args.soft_match)

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
