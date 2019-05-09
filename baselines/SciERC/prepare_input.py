import spacy
import json
import argparse

nlp = spacy.load('en_core_sci_sm')


def int_overlap(a1, b1, a2, b2):
    """Checks whether two intervals overlap"""
    if b1 < a2 or b2 < a1:
        return False
    return True


class Token:
    def __init__(self, text, start_char, end_char):
        self.text = text
        self.start_char = start_char
        self.end_char = end_char


class Sentence:
    def __init__(self, data):
        self.data = data
        self.tokens = []
        self.whitespaces = []
        self.sentence = []
        self.tokenize()
        self.ner_list = []
        self.ent_ids_to_ner_ids = {}
        self.get_entities()
        self.clusters = []
        self.get_clusters()
        self.relations = []
        self.get_relations()

    def tokenize(self):
        sent = self.data['text']
        doc = nlp(sent, disable=['tagger', 'parser', 'ner', 'textcat'])
        char_pos = 0
        for token in doc:
            end_pos = char_pos + len(token.text) - 1
            self.sentence.append(token.text)
            self.tokens.append(Token(token.text, char_pos, end_pos))
            self.whitespaces.append(token.whitespace_)
            char_pos = end_pos + 1
            if token.whitespace_:
                char_pos += 1

    def get_entities(self):
        for ent_id, ent in enumerate(self.data['entities']):
            if not ent['is_mentioned']:
                continue

            ent_type = ent['label']

            for entity in ent['names'].values():
                if not entity['is_mentioned']:
                    continue

                for mention in entity['mentions']:
                    start = mention[0]
                    end = mention[1] - 1
                    ne_start = None
                    ne_end = None
                    for idx, token in enumerate(self.tokens):
                        if int_overlap(start, end,
                                       token.start_char, token.end_char):
                            if ne_start is None:
                                ne_start = idx
                            ne_end = idx

                    ne = (ne_start, ne_end, ent_type)
                    if ne not in self.ner_list:
                        self.ner_list.append(ne)
                        ner_id = len(self.ner_list) - 1
                    else:
                        ner_id = self.ner_list.index(ne)

                    if ent_id not in self.ent_ids_to_ner_ids:
                        self.ent_ids_to_ner_ids[ent_id] = set([])
                    self.ent_ids_to_ner_ids[ent_id].add(ner_id)

    def get_clusters(self):
        for ent_id in self.ent_ids_to_ner_ids:
            nes = self.ent_ids_to_ner_ids[ent_id]
            if len(nes) < 2:
                continue
            cluster = []
            for ne_id in nes:
                cluster.append(self.ner_list[ne_id][:2])
            self.clusters.append(cluster)

    def get_relations(self):
        for info in self.data['interactions']:
            ent_id_1 = info['participants'][0]
            ent_id_2 = info['participants'][1]
            rel_type = [info['type']]

            for ne_id_1 in self.ent_ids_to_ner_ids[ent_id_1]:
                for ne_id_2 in self.ent_ids_to_ner_ids[ent_id_2]:
                    ne_1 = list(self.ner_list[ne_id_1][:2])
                    ne_2 = list(self.ner_list[ne_id_2][:2])
                    if (ne_1 + ne_2 + rel_type) not in self.relations:
                        self.relations.append(ne_1 + ne_2 + rel_type)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--text_mode', action='store_true', default=False)
    parser.add_argument('--output', required=True)
    parser.add_argument('--output_whitespaces', default=None)
    args = parser.parse_args()

    input_fname = args.input
    output_fname = args.output
    whitespaces = {}

    if args.text_mode:
        with open(input_fname) as f:
            data = f.readlines()
    else:
        with open(input_fname) as f:
            data = json.load(f)

    with open(output_fname, 'w') as f:
        for d in data:
            if args.text_mode:
                id, text = d.split('\t')
                d = {
                    "id": id,
                    "text": text,
                    "entities": {},
                    "interactions": []
                }
            doc_key = d['id']
            s = Sentence(d)
            whitespaces[doc_key] = s.whitespaces
            scierc_d = dict(doc_key=doc_key, sentences=[s.sentence],
                            ner=[s.ner_list], relations=[s.relations],
                            clusters=s.clusters)
            f.write(json.dumps(scierc_d) + '\n')

    whitespaces_fname = args.output_whitespaces
    if whitespaces_fname:
        with open(whitespaces_fname, 'w') as f:
            json.dump(whitespaces, f, indent=True)


if __name__ == '__main__':
    main()
