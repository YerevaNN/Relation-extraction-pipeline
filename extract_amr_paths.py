import argparse
import io
import json
import penman
import re
from nltk.metrics.distance import edit_distance
from tqdm import tqdm

def nodes_from_word(graph, word):
    #  SOFT MATCHING
    ed_threshold = 1.
    nodes = [(edit_distance(str(attr.target), word), attr)
             for attr in graph.attributes()]
    nodes = sorted(nodes)   
    sources = [node[1].source for node in nodes 
               if node[0] == nodes[0][0] and len(word) * ed_threshold > node[0]]
    
    return sources
        

def remove_quotes(word):
    if isinstance(word, str) and word[0] == '"' and word[-1] == '"':
        return word[1:-1]
    return word

def word_from_node(graph, node):
    for attr in graph.attributes():
        if attr.target == 'name':
            continue
        if attr.source == node:
            return remove_quotes(attr.target)

    for attr in graph.attributes():
        if attr.source == node:
            return remove_quotes(attr.target)
    return None

def reconstruct_path(back_edges, node_1, node_2):
    """ Assumes there are no loops in 'back_edges' """
    path = []
    curr_node = node_2
    while curr_node and curr_node != node_1:
        path.append(back_edges[curr_node])
        curr_node = back_edges[curr_node].source
    return path

def find_path(graph, node_1, node_2):
    """ Retrun list of edges from node_1 to node_2 """
    
    edges = graph.edges()
    q = [node_1]
    visited_nodes = [node_1]
    back_edges = {}
    while q:
        n = q.pop(0)
        for e in edges:
            source = e.source
            target = e.target
            if source != n or target in visited_nodes:
                continue
            q.append(target)
            visited_nodes.append(target)
            back_edges[target] = e
            if target == node_2:
                return reconstruct_path(back_edges, node_1, node_2)
    return None

def LCA(graph, node_1, node_2):

    """ Returns the name of the lowest node, that is an ancestor for both
        node_1 and node_2"""
    edges = graph.edges()
    nodes = graph.variables()
    q_1 = [node_1]
    q_2 = [node_2]
    visited_edges_1 = [node_1]
    visited_edges_2 = [node_2]
    
    while q_1 or q_2:
        if q_1:
            n_1 = q_1.pop(0)
            for e in edges:
                if e.target != n_1:
                    continue
                s = e.source
                if s in visited_edges_2:
                    return s
                if s in visited_edges_1:
                    continue
                visited_edges_1.append(s)
                q_1.append(s)
        if q_2:
            n_2 = q_2.pop(0)
            for e in edges:
                if e.target != n_2:
                    continue
                s = e.source
                if s in visited_edges_1:
                    return s
                if s in visited_edges_2:
                    continue
                visited_edges_2.append(s)
                q_2.append(s)
    return None

def paths_for_words(graph, word_1, word_2):

    paths = []
    nodes_1 = nodes_from_word(graph, word_1)
    nodes_2 = nodes_from_word(graph, word_2)
    for node_1 in nodes_1:
        for node_2 in nodes_2:
            ancestor = LCA(graph, node_1, node_2)
            if ancestor:
                paths.append([find_path(graph, ancestor, node_1),
                              find_path(graph, ancestor, node_2)])
    return paths

def sentence_from_path(graph, path):
    sentence = []
    if path:
        for edge in path:
            sentence.append((word_from_node(graph, edge.source), edge.relation))
    return sentence    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', required=True,
                        help='Input file', type=str)
    parser.add_argument('--output', '-o', required=True,
                        help='Output file', type=str)
    args = parser.parse_args()

    # input JSON is expected to have tuples of following type:
    #             (id, amr, (word1, word2))

    with io.open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)
    output_paths = []
    l = []
    for id, amr, words in tqdm(data):
        try:
            amr_parsed = amr
            while amr_parsed != re.sub(r'"([^~"]*)"+([^~]*)"~', '"\\1\\2"~', amr_parsed):
                amr_parsed = re.sub(r'"([^~"]*)"+([^~]*)"~', '"\\1\\2"~', amr_parsed)
            amr_parsed = re.sub(r"\~e\.[0-9,]+", "", amr_parsed)  # Removing alignment tags
            # amr_parsed = re.sub(r'"{2,}', '""', amr_parsed)  # Removing """"" this kind of things
            graph = penman.decode(amr_parsed)
        except:
            print(amr)
            print(amr_parsed)
            raise Exception("AMR can not be parsed by PenMan")
        paths = paths_for_words(graph, words[0], words[1])
        if paths:
            for path in paths:
                sentences = [[(word_from_node(graph, nodes_from_word(graph, words[0])[0]), None)],
                             [(word_from_node(graph, nodes_from_word(graph, words[1])[0]), None)]]
                sentences[0] += sentence_from_path(graph, path[0])
                sentences[1] += sentence_from_path(graph, path[1])
                output_paths.append((id, sentences))
        else:
            output_paths.append((id, [[], []])) 

    with io.open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_paths, f, indent=True)

if __name__ == '__main__':
    main()

