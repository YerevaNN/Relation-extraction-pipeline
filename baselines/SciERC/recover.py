import json
import argparse


def join_with_ws(tokens, ws):
    joined = ''
    for token, whitespace in zip(tokens[:-1], ws[:-1]):
        joined += token
        joined += whitespace

    return joined + tokens[-1]


def mention_tokens_to_chars(mention, tokens, whitespaces):
    token_start = mention[0]
    token_end = mention[1]
    token_lens = [len(t) for t in tokens]
    ws_lens = [len(w) for w in whitespaces]
    char_start = sum(token_lens[:token_start]) + sum(ws_lens[:token_start])
    char_end = sum(token_lens[:token_end + 1]) + sum(ws_lens[:token_end]) - 1
    return char_start, char_end


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction', required=True)
    parser.add_argument('--scierc_input', required=True)
    parser.add_argument('--whitespaces', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    prediction_fname = args.prediction
    original_fname = args.scierc_input
    whitespace_fname = args.whitespaces
    output_fname = args.output
    sents = []
    original_sents = []

    with open(prediction_fname) as f:
        for line in f:
            sents.append(json.loads(line))
    with open(original_fname) as f:
        for line in f:
            original_sents.append(json.loads(line))
    with open(whitespace_fname) as f:
        whitespaces = json.load(f)

    data = []
    absent_entities_in_coref = 0
    absent_entities_in_rel = 0
    for s, o_s in zip(sents, original_sents):
        ws = whitespaces[s['doc_key']]
        text = o_s['sentences'][0]
        new_s = {
            'id': s['doc_key'],
            'text': join_with_ws(text, ws)
        }
        entities = {}
        for ner in s['ner'][0]:
            start = ner[0]
            end = ner[1]
            ent_type = ner[2]
            t = join_with_ws(text[start:end + 1], ws[start:end + 1])
            entities[(start, end)] = {
                'text': t,
                'type': ent_type
            }

        # add entities from relations
        # because SciERC relation extractor can produce entities which were not detected by NER
        missing_entities = set()

        for cluster in s['coref']:
            for ent in cluster:
                if tuple(ent) not in entities:
                    missing_entities.add(tuple(ent))
        for rel in s['relation'][0]:
            s1 = rel[0]
            e1 = rel[1]
            s2 = rel[2]
            e2 = rel[3]
            if (s1, e1) not in entities:
                missing_entities.add((s1, e1))
            if (s2, e2) not in entities:
                missing_entities.add((s2, e2))

        for start, end in missing_entities:
            t = join_with_ws(text[start:end + 1], ws[start:end + 1])
            entities[(start, end)] = {
                'text': t,
                'type': 'unknown'
            }

        ent_group_num = 0
        for cluster in s['coref']:
            for ent in cluster:
                if tuple(ent) not in entities:
                    absent_entities_in_coref += 1
                    continue
                entities[tuple(ent)]['cluster'] = ent_group_num
            ent_group_num += 1

        new_s['unique_entities'] = {}
        for k in entities:
            if 'cluster' not in entities[k]:
                entities[k]['cluster'] = ent_group_num
                ent_group_num += 1

            c = entities[k]['cluster']
            ent_name = entities[k]['text']
            ent_type = entities[k]['type']

            if c not in new_s['unique_entities']:
                new_s['unique_entities'][c] = {
                    'versions': {},
                    'labels_major': [ent_type]
                }

            char_mention = mention_tokens_to_chars(k, text, ws)
            if ent_name not in new_s['unique_entities'][c]['versions']:
                new_s['unique_entities'][c]['versions'][ent_name] = {
                    'labels': [ent_type],
                    'labels_major': [ent_type],
                    'mentions': [char_mention]
                }
            else:
                new_s['unique_entities'][c]['versions'][ent_name][
                    'mentions'].append(char_mention)

        new_s['extracted_information'] = []
        for rel in s['relation'][0]:
            s1 = rel[0]
            e1 = rel[1]
            s2 = rel[2]
            e2 = rel[3]
            if (s1, e1) not in entities or (s2, e2) not in entities:
                absent_entities_in_rel += 1
                continue
            int_type = rel[4]
            c1 = entities[(s1, e1)]['cluster']
            c2 = entities[(s2, e2)]['cluster']
            name_1 = entities[(s1, e1)]['text']
            name_2 = entities[(s2, e2)]['text']
            new_s['extracted_information'].append({
                'contains_implicit_entity': False,
                'label': 1,
                'participant_ids': [
                    c1,
                    c2
                ],
                'participant_a': name_1,
                'participant_b': name_2,
                'interaction_type': int_type
            })
        data.append(new_s)

    print('Number of absent entities in coreference groups: ',
          absent_entities_in_coref)
    print('Number of absent entities in relations: ', absent_entities_in_rel)

    with open(output_fname, 'w') as f:
        json.dump(data, f, indent=True)


if __name__ == '__main__':
    main()
