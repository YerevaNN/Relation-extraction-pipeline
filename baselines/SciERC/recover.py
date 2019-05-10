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
    char_end = sum(token_lens[:token_end + 1]) + sum(ws_lens[:token_end])
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
    for s, o_s in zip(sents, original_sents):
        ws = whitespaces[s['doc_key']]
        new_s = {
            'id': s['doc_key']
        }
        text = o_s['sentences'][0]
        entities = {}
        if 'ner' in s:
            for ner in s['ner'][0]:
                start = ner[0]
                end = ner[1]
                ent_type = ner[2]
                t = join_with_ws(text[start:end + 1], ws[start:end + 1])
                entities[(start, end)] = {
                    'text': t,
                    'type': ent_type
                }

        # Adding entities from clusters and relations because SciERC relation
        # extraction and coreference resolution can produce entity spans which
        # were not detected by NER
        missing_entities = set()
        if 'coref' in s:
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
        if 'coref' in s:
            for cluster in s['coref']:
                empty_cluster = True
                for ent in cluster:
                    if 'cluster' not in entities[tuple(ent)]:
                        entities[tuple(ent)]['cluster'] = ent_group_num
                        empty_cluster = False
                if not empty_cluster:
                    ent_group_num += 1

        new_s['entities'] = {}
        for k in entities:
            if 'cluster' not in entities[k]:
                entities[k]['cluster'] = ent_group_num
                ent_group_num += 1

            c = entities[k]['cluster']
            ent_name = entities[k]['text']
            ent_type = entities[k]['type']

            if c not in new_s['entities']:
                new_s['entities'][c] = {
                    'names': {},
                    'label': ent_type
                }

            char_mention = mention_tokens_to_chars(k, text, ws)
            if ent_name not in new_s['entities'][c]['names']:
                new_s['entities'][c]['names'][ent_name] = {
                    'label': ent_type,
                    'mentions': [char_mention],
                    'is_mentioned': True
                }
            else:
                new_s['entities'][c]['names'][ent_name]['mentions'].append(
                    char_mention)
        new_s['entities'] = [new_s['entities'][i] for i in range(ent_group_num)]

        new_s['interactions'] = []
        for rel in s['relation'][0]:
            s1 = rel[0]
            e1 = rel[1]
            s2 = rel[2]
            e2 = rel[3]
            int_type = rel[4]
            c1 = entities[(s1, e1)]['cluster']
            c2 = entities[(s2, e2)]['cluster']
            new_s['interactions'].append({
                'label': 1,
                'participants': [
                    c1,
                    c2
                ],
                'type': int_type
            })
        data.append(new_s)

    with open(output_fname, 'w') as f:
        json.dump(data, f, indent=True)


if __name__ == '__main__':
    main()
