import os
import logging
import argparse
import json
import pickle
import spacy
from spacy.tokens import Doc, Token
from typing import List, Tuple
import networkx as nx
import copy
import numpy as np
import random


def edge_list_to_graph(edge_list):
    graph = nx.Graph()
    index = {}
    for edge in edge_list:
        n1 = edge[3]
        if n1 not in index:
            index[n1] = len(index)
        n1 = index[n1]
        n2 = edge[6]
        if n2 not in index:
            index[n2] = len(index)
        n2 = index[n2]
        if n1 not in graph:
            graph.add_node(n1, label=edge[2])
        if n2 not in graph:
            graph.add_node(n2, label=edge[5])
        graph.add_edge(n1, n2, label=edge[4])
    return graph


class SentenceSpan(object):
    def __init__(self, sid: int, begin_char: int, end_char: int):
        self.sid: int = sid
        """sentence counter within document"""
        self.begin_char: int = begin_char
        """begin character offset of sentence within document"""
        self.end_char: int = end_char
        """end character offset of sentence within document"""


class Qrel(object):
    def __init__(self, qedge_list: List[Tuple], docid: int, ans_sent_id: int):
        self.qedge_list: List[Tuple] = qedge_list
        """Edge list of query dependency graph, could be tree or forest
        in case of multi-sentence queries."""
        self.docid: int = docid
        """Document ID where query is answered."""
        self.ans_sent_id: int = ans_sent_id
        """Within this docid, which sentence has gold response for query."""


class SquadParser(object):
    """Converts SQuAD data into query graphs and corpus sentence graphs using
    a dependency parse."""

    def __init__(self, av):
        self.av = av
        self.nlp = spacy.load("en_core_web_sm")

    # def __del__(self):

    def find(self, sent_spans: List[SentenceSpan],
             qbegin_char: int, qend_char: int):
        """
        :param sent_spans: list of sentence spans in a document
        :param qbegin_char: begin char offset of gold sentence
        :param qend_char: end char offset of gold sentence
        :return: sentence ID within document
        """
        sent_span: SentenceSpan
        for sent_span in sent_spans:
            if sent_span.end_char < qbegin_char:
                continue
            if sent_span.begin_char > qend_char:
                continue
            return sent_span.sid
        return None

    def collect_graph(self, docid: int, doc: Doc):
        sent_spans: List[SentenceSpan] = list()
        edge_list: List[Tuple] = list()  # edge tuples have str and int
        num_sent = -1
        graph_list = []
        for sent in doc.sents:
            sent_edge_list = []
            num_sent += 1
            # print(num_sent, sent.start_char, sent.end_char, sent)
            sent_spans.append(SentenceSpan(num_sent, sent.start_char,
                                           sent.end_char))
            for tok in sent:
                tok: Token = tok
                # tok.i is the token offset within sentence
                # (doc_id, num_sent, tok.i) is usable as global node ID
                # tok.text is the string form of the token
                # tok.lemma is stemmed form
                # tok.dep_ is label of edge to dependency parent
                # tok.head.i is token offset of parent in sentence
                # (num_sent, tok.head.i) is usable as global node ID
                row = (docid, num_sent, str(tok.text), int(tok.i),
                       str(tok.dep_), str(tok.head), int(tok.head.i))
                sent_edge_list.append(row)
                edge_list.append(row)
                msg = "d=%d s=%d %s_%d %s %s_%d" % row
                logging.debug(msg)
            graph = edge_list_to_graph(sent_edge_list)
            graph_list.append(graph)
            # How to record provenance to trace back from graphs to SQuAD?
        return sent_spans, edge_list, graph_list

    def run_tests(self):
        for px, passage in enumerate([
            "Obama was born in Hawaii.  He was elected president in 2008.",
            "Mechanical clocks lose or gain time, unless regulated often.",
            "Boil the potato.  Add chopped onions and stir-fry."
        ]):
            print(passage)
            doc = self.nlp(passage)
            self.collect_graph(px, doc)

    def parse_squad(self, min_query_nodes, max_query_nodes, min_corpus_nodes, max_corpus_nodes, neg_to_pos_ratio):
        import time, tqdm
        qrelss: List[Qrel] = list()
        docid = -1
        t_begin = time.time()
        with open(os.path.expanduser(self.av.squad), "rb") as squad_file:
            squad_train = json.load(squad_file)["data"]
        num_sentences = []
        query_graphs = []
        corpus_graphs = []
        for rx in tqdm.tqdm(range(len(squad_train))):
            article_query_graphs = []
            article_corpus_graphs = []
            rec = squad_train[rx]
            for para in rec["paragraphs"]:
                cdoc: Doc = self.nlp(para["context"])
                docid += 1
                sent_spans, edge_list, graph_list = self.collect_graph(docid, cdoc)
                num_sentences.append(len(graph_list))
                for qa in para["qas"]:
                    qdoc = self.nlp(qa["question"])
                    # If the query breaks up into more than one dependency
                    # graphs, we should collect the pieces together as the
                    # query graph.
                    _, qedge_list, qgraph_list = self.collect_graph(docid, qdoc)
                    look = None
                    for ans in qa["answers"]:
                        ans_start = int(ans["answer_start"])
                        ans_len = len(ans["text"])
                        # print(ans_start, ans_len, type(para["context"]))
                        span = para["context"][ans_start: ans_start + ans_len]
                        # print(ans["text"], span, ans["answer_start"])
                        #  , ans["is_impossible"]
                        look = self.find(sent_spans, ans_start,
                                         ans_start + ans_len)
                        if look is not None:
                            qrel = Qrel(qedge_list=qedge_list,
                                        docid=docid, ans_sent_id=look)
                            qrelss.append(qrel)
                    if len(qgraph_list) != 1 or len(qa["answers"]) != 1 or qa['is_impossible'] or look is None or \
                            len(qgraph_list[0].nodes()) < min_query_nodes or len(qgraph_list[0].nodes()) > max_query_nodes or \
                            len(graph_list[look].nodes()) < min_corpus_nodes or len(graph_list[look].nodes()) > max_corpus_nodes:
                        logging.debug(f"Skipping question: {qa['question']}")
                    else:
                        article_query_graphs.append(qgraph_list[0])
                        article_corpus_graphs.append(copy.deepcopy(graph_list[look]))
            if len(article_query_graphs) < (neg_to_pos_ratio+1):
                continue
            for i in range(len(article_query_graphs)):
                query_graphs.append(article_query_graphs[i])
                negative_corpus_graphs = article_corpus_graphs[:i]+article_corpus_graphs[i+1:]
                negative_corpus_graphs = random.sample(negative_corpus_graphs, neg_to_pos_ratio)
                cg = [article_corpus_graphs[i]] + negative_corpus_graphs
                corpus_graphs.append(cg)
        num_query_graphs = len(query_graphs)
        labels = np.concatenate([np.ones([num_query_graphs,1]), np.zeros([num_query_graphs,neg_to_pos_ratio])], axis=1)
        t_end = time.time()
        logging.info("time taken %g", t_end - t_begin)
        logging.info("number_of_queries %d", len(query_graphs))
        params = {
            'max_query_nodes': max_query_nodes,
            'max_corpus_nodes': max_corpus_nodes,
            'corpus_per_query': 1,
            'total_corpus_per_query': neg_to_pos_ratio,
            'val_fraction': 1.0,
            'test_fraction': 1.0,
        }
        third = int(num_query_graphs/3)
        query_graphs_train = query_graphs[:third]
        corpus_graphs_train = corpus_graphs[:third]
        labels_train = labels[:third]
        query_graphs_val = query_graphs[third:2*third]
        corpus_graphs_val = corpus_graphs[third:2*third]
        labels_val = labels[third:2*third]
        query_graphs_test = query_graphs[2*third:]
        corpus_graphs_test = corpus_graphs[2*third:]
        labels_test = labels[2*third:]

        with open(self.av.data, "wb") as data_file:
            pickle.dump([params, query_graphs_train, corpus_graphs_train, query_graphs_test, corpus_graphs_test,
                         query_graphs_val, corpus_graphs_val, labels_train, labels_test, labels_val], data_file)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--squad", help="/path/to/squad/corpus.json",
                    default="../product_graph_approach/data/squad/train-v2.0.json")
    ap.add_argument("--data", help="/path/to/squad/corpus.json",
                    default="../product_graph_approach/data/squad/squad.pkl")
    ap.add_argument("--loglevel", help="log level",
                    choices=["DEBUG", "INFO", "WARN"], default="INFO")
    av = ap.parse_args()
    numeric_level = getattr(logging, av.loglevel.upper(), None)
    logging.basicConfig(filename='../product_graph_approach/logs/squad.log', level=numeric_level,
                        format='%(asctime)s - %(levelname)s: %(message)s')
    squad_parser = SquadParser(av)
    # squad_parser.run_tests()
    squad_parser.parse_squad(min_query_nodes=4, max_query_nodes=10, min_corpus_nodes=10, max_corpus_nodes=20,
                             neg_to_pos_ratio=9)


if __name__ == "__main__":
    main()
