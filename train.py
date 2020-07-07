import logging
import os
import pickle as pkl
import random
import time
from argparse import ArgumentParser
import json

import networkx as nx
import numpy as np
import tensorflow as tf
from scipy import sparse

from model import RandomWalkModel
from random_walk import build_product_graph_and_sample
from utils import train, evaluate, indices_to_one_hot, word_embedding

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
numeric_label = {}
word_embedding_dict = {}


def dep_graph_features(graph, max_nodes):
    global numeric_label, word_embedding_dict
    n = len(graph.nodes())
    assert (n <= max_nodes)
    dim_node_features = 300
    dim_edge_features = 46
    node_features = np.zeros((max_nodes, 1, dim_node_features))
    node_labels = nx.get_node_attributes(graph, 'label')
    for i in range(0, n):
        if node_labels[i] not in word_embedding_dict:
            word_embedding_dict[node_labels[i]] = word_embedding(node_labels[i])
        node_features[i, 0, :] = np.array(word_embedding_dict[node_labels[i]])

    edge_features = np.zeros((max_nodes, max_nodes, dim_edge_features))
    mask = nx.adjacency_matrix(graph).todense()
    for i in range(0, n):
        for j in range(0, n):
            if mask[i, j] == 1:
                edge_label = graph[i][j]['label']
                if edge_label not in numeric_label:
                    numeric_label[edge_label] = len(numeric_label)
                    assert(len(numeric_label) <= 46)
                edge_features[i, j, :] = indices_to_one_hot(numeric_label[edge_label], dim_edge_features)

    return node_features, edge_features


def synth_graph_features(graph, max_nodes, num_node_labels, num_edge_labels, one_hot):
    n = len(graph.nodes())
    assert (n <= max_nodes)
    dim_node_features = 1
    dim_edge_features = 1
    if one_hot:
        dim_node_features = num_node_labels
        dim_edge_features = num_edge_labels
    node_features = np.zeros((max_nodes, 1, dim_node_features))
    node_labels = nx.get_node_attributes(graph, 'label')
    for i in range(0, n):
        if i in node_labels:
            if one_hot:
                node_features[i, 0, :] = indices_to_one_hot(int(node_labels[i]), num_node_labels)
            else:
                node_features[i, 0, 0] = node_labels[i]

    edge_features = np.zeros((max_nodes, max_nodes, dim_edge_features))
    mask = nx.adjacency_matrix(graph).todense()
    for i in range(0, n):
        for j in range(0, n):
            if mask[i, j] == 1:
                if one_hot:
                    edge_features[i, j, :] = indices_to_one_hot(int(graph[i][j]['label']), num_edge_labels)
                else:
                    edge_features[i, j, 0] = graph[i][j]['label']

    return node_features, edge_features


def clevr_graph_features(graph, max_nodes):
    n = len(graph.nodes())
    assert (n <= max_nodes)

    dim_node_features = len(graph.nodes()[0]['label'])
    dim_edge_features = len(list(graph.edges(data=True))[0][2]['label'])
    node_features = np.zeros((max_nodes, 1, dim_node_features))
    node_labels = nx.get_node_attributes(graph, 'label')
    for i in range(0, n):
        if i in node_labels:
            node_features[i, 0, :] = node_labels[i]

    edge_features = np.zeros((max_nodes, max_nodes, dim_edge_features))
    mask = nx.adjacency_matrix(graph).todense()
    for i in range(0, n):
        for j in range(0, n):
            if mask[i, j] == 1:
                edge_features[i, j, :] = graph[i][j]['label']

    return node_features, edge_features


def features(graph, dataset, max_nodes, **kwargs):
    if dataset == "synthetic":
        assert (kwargs['one_hot'] is not None)
        assert (kwargs['num_node_labels'] is not None)
        assert (kwargs['num_edge_labels'] is not None)
        return synth_graph_features(graph, max_nodes, kwargs['num_node_labels'], kwargs['num_edge_labels'],
                                    kwargs['one_hot'])
    elif dataset == "squad":
        return dep_graph_features(graph, max_nodes)
    elif dataset == "clevr":
        return clevr_graph_features(graph, max_nodes)


def corpus_data(corpus_graphs, query_graphs, num_walks, max_length_walk, restart_prob, labels, num_queries, data_params,
                one_hot, walk_method, dataset):
    global word_embedding_dict
    if not corpus_graphs or not corpus_graphs[0]:
        return None

    walks = []
    corpus_node_features = [None] * num_queries
    corpus_edge_features = [None] * num_queries
    for i in range(num_queries):
        corpus_node_features[i] = []
        corpus_edge_features[i] = []
        for corpus_graph in corpus_graphs[i]:
            node_features, edge_features = features(corpus_graph, dataset, data_params['max_corpus_nodes'],
                                                    num_edge_labels=data_params.get('edge_types'),
                                                    num_node_labels=data_params.get('node_types'), one_hot=one_hot)
            corpus_node_features[i].append(node_features)
            corpus_edge_features[i].append(edge_features)
        max_product_graph_nodes = data_params['max_query_nodes'] * data_params['max_corpus_nodes']
        walk = build_product_graph_and_sample(query_graphs[i], corpus_graphs[i], num_walks, max_length_walk,
                                              restart_prob, walk_method, max_product_graph_nodes, word_embedding_dict)
        walk_shape = walk.shape
        walk = sparse.csr_matrix(walk.reshape((-1)))
        # padded_walk = np.zeros((walk_shape[0], walk_shape[1], walk_shape[2], max_product_graph_nodes,
        #                         max_product_graph_nodes))
        # padded_walk[:, :, :, :walk_shape[3], :walk_shape[4]] = walk
        # padded_walk = sparse.csr_matrix(padded_walk.reshape((-1)))
        walks.append(walk)

        corpus_node_features[i] = np.stack(corpus_node_features[i], axis=0)
        corpus_edge_features[i] = np.stack(corpus_edge_features[i], axis=0)
        corpus_edge_features_shape = corpus_edge_features[i].shape
        corpus_edge_features[i] = sparse.csr_matrix(corpus_edge_features[i].reshape((-1)))

    return [corpus_node_features, corpus_edge_features, walks, labels, corpus_edge_features_shape, walk_shape]


def query_data(query_graphs, num_queries, data_params, one_hot, dataset):
    query_features = []
    for i in range(num_queries):
        query_graph = query_graphs[i]
        node_features, edge_features = features(query_graph, dataset, data_params['max_query_nodes'],
                                                num_edge_labels=data_params.get('edge_types'),
                                                num_node_labels=data_params.get('node_types'), one_hot=one_hot)
        query_features.append([node_features, edge_features])
    return query_features


def load_data_and_random_walk_sample(data_path, num_walks, max_length_walk, restart_prob, walk_method, num_queries,
                                     one_hot, dataset):
    if dataset == "synthetic" or dataset == "clevr":
        pickle_in = open(data_path, "rb")
        data_params, query_graphs, train_corpus, test_corpus, validation_corpus, train_labels, test_labels, \
            validation_labels = pkl.load(pickle_in)
        pickle_in.close()
        query_features = query_data(query_graphs, num_queries, data_params, one_hot, dataset)

        train_corpus_data = corpus_data(train_corpus, query_graphs, num_walks, max_length_walk, restart_prob,
                                        train_labels, num_queries, data_params, one_hot, walk_method, dataset)
        validation_corpus_data = corpus_data(validation_corpus, query_graphs, num_walks, max_length_walk, restart_prob,
                                             validation_labels, num_queries, data_params, one_hot, walk_method, dataset)
        test_corpus_data = corpus_data(test_corpus, query_graphs, num_walks, max_length_walk, restart_prob, test_labels,
                                       num_queries, data_params, one_hot, walk_method, dataset)

        params = dict(
            num_nodes_query=data_params['max_query_nodes'],
            num_nodes_corpus=data_params['max_corpus_nodes'],
            node_types=data_params['node_types'] if 'node_types' in data_params else None,
            edge_types=data_params['edge_types'] if 'edge_types' in data_params else None,
        )
        return params, query_features, train_corpus_data, query_features, validation_corpus_data, query_features, test_corpus_data
    elif dataset == "squad":
        pickle_in = open(data_path, "rb")
        data_params, query_graphs_train, train_corpus, query_graphs_test, test_corpus, query_graphs_val, \
            validation_corpus, train_labels, test_labels, validation_labels = pkl.load(pickle_in)
        pickle_in.close()
        query_features_train = query_data(query_graphs_train, num_queries[0], data_params, one_hot, dataset)
        query_features_val = query_data(query_graphs_val, num_queries[1], data_params, one_hot, dataset)
        query_features_test = query_data(query_graphs_test, num_queries[1], data_params, one_hot, dataset)

        train_corpus_data = corpus_data(train_corpus, query_graphs_train, num_walks, max_length_walk, restart_prob,
                                        train_labels, num_queries[0], data_params, one_hot, walk_method, dataset)
        validation_corpus_data = corpus_data(validation_corpus, query_graphs_val, num_walks, max_length_walk,
                                             restart_prob, validation_labels, num_queries[1], data_params, one_hot,
                                             walk_method, dataset)
        test_corpus_data = corpus_data(test_corpus, query_graphs_test, num_walks, max_length_walk, restart_prob,
                                       test_labels, num_queries[1], data_params, one_hot, walk_method, dataset)

        params = dict(
            num_nodes_query=data_params['max_query_nodes'],
            num_nodes_corpus=data_params['max_corpus_nodes'],
        )
        return params, query_features_train, train_corpus_data, query_features_val, validation_corpus_data, \
            query_features_test, test_corpus_data


def main():
    params = {}
    ap = ArgumentParser()
    ap.add_argument("--data_path", type=str, default="./data/squad/squad.pkl")
    ap.add_argument("--dataset", type=str, default="squad", choices=["synthetic", "squad", "clevr"])
    ap.add_argument("--logfile", type=str, default="./logs/gx-net.log")
    ap.add_argument("--name", type=str, default="model")
    ap.add_argument("--num_queries", type=int, default=None)
    ap.add_argument("--num_queries_train", type=int, default=10)
    ap.add_argument("--num_queries_eval", type=int, default=10)
    ap.add_argument("--num_walks", type=int, default=10)
    ap.add_argument("--max_length_walk", type=int, default=10)
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--early_stopping", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--restart_prob", type=float, default=0.0)
    ap.add_argument("--nlayer1", type=int, default=5)
    ap.add_argument("--nlayer2", type=int, default=5)
    ap.add_argument("--elayer1", type=int, default=5)
    ap.add_argument("--elayer2", type=int, default=5)
    ap.add_argument("--nelayer1", type=int, default=10)
    ap.add_argument("--nelayer2", type=int, default=10)
    ap.add_argument("--dropout", type=float, default=1.0)
    ap.add_argument("--one_hot", type=bool, default=False)
    ap.add_argument("--sparse_walk", type=bool, default=True)
    ap.add_argument("--walk_method", type=str, default="random", choices=["random", "biased"])
    ap.add_argument("--num_threads", type=int, default=1)
    av = ap.parse_args()
    params.update(vars(av))
    tf.set_random_seed(av.seed)
    random.seed(av.seed)
    np.random.seed(av.seed)
    os.environ['PYTHONHASHSEED'] = str(av.seed)
    logger = logging.getLogger('logger')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s', filename=av.logfile,
                        filemode='a')
    logging.info(f'training code for the params:\n{json.dumps(params, indent=4)}')
    if av.num_queries is not None:
        num_queries = av.num_queries
        av.num_queries_train = num_queries
        av.num_queries_eval = num_queries
    else:
        num_queries = (av.num_queries_train, av.num_queries_eval)

    params, query_features_train, corpus_train, query_features_val, corpus_val, query_features_test, corpus_test = \
        load_data_and_random_walk_sample(av.data_path, av.num_walks, av.max_length_walk, av.restart_prob,
                                         av.walk_method, num_queries, av.one_hot, av.dataset)
    from utils import word_found, word_not_found
    logging.info(f"embeddings: found={word_found}, not found={word_not_found}")
    dim_node_features = query_features_train[0][0].shape[2]
    dim_edge_features = query_features_train[0][1].shape[2]
    num_nodes_corpus = params['num_nodes_corpus']
    sparse_walk = av.sparse_walk
    model = RandomWalkModel(num_nodes_query=params['num_nodes_query'], num_nodes_corpus=params['num_nodes_corpus'],
                            dim_node_features=dim_node_features, dim_edge_features=dim_edge_features,
                            node_layer1=av.nlayer1, node_layer2=av.nlayer2, edge_layer1=av.elayer1,
                            edge_layer2=av.elayer2, layer1=av.nelayer1, layer2=av.nelayer2, name=av.name,
                            delta=av.delta, dropout=av.dropout, sparse_walk=sparse_walk, walk_shape=corpus_train[5])
    # logging.info(f"number of parameters: {np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])}")
    # tf.reset_default_graph()
    sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=av.num_threads))
    sess.run(tf.global_variables_initializer())
    # writer = tf.summary.FileWriter(LOGDIR)
    # writer.add_graph(sess.graph)
    # summary_op = tf.summary.merge_all()
    # fig.show()
    # fig.canvas.draw()
    epoch_list = []
    train_loss_list = []
    train_map_list = []
    val_loss_list = []
    val_map_list = []
    # model.load(sess)
    cur_time = time.time()
    for cur_epoch in range(av.epochs):
        t = time.time()
        num_corpus = corpus_train[0][0].shape[0]
        train_map, train_mean_loss, average_precision_at_k, train_mrr = \
            train(model, sess, query_features_train, corpus_train, av.num_queries_train, sparse_walk)
        train_loss_list.append(train_mean_loss)
        train_map_list.append(train_map)

        # model.save(sess)
        epoch_list.append(cur_epoch)
        val_map, val_mean_loss, average_precision_at_k, val_mrr, _, _ = \
            evaluate(model, sess, query_features_val, corpus_val, av.num_queries_eval, sparse_walk)
        # logging.info(f"average precision at k: {average_precision_at_k}")
        val_loss_list.append(val_mean_loss)
        val_map_list.append(val_map)

        logging.info(f"epoch:\t{cur_epoch},\tval mean loss: {val_mean_loss:.5f},"
                     f"\tval mean average precision: {val_map:.5f},"
                     f"\tval mean reciprocal rank: {val_mrr:.5f},"
                     f"\ttrain mean loss: {train_mean_loss:.5f},"
                     f"\ttrain mean average precision: {train_map:.5f},"
                     f"\ttrain mean reciprocal rank: {train_mrr:.5f},"
                     f"\ttime: {(time.time() - t):.2f}")
        if cur_epoch > av.early_stopping and val_loss_list[-1] > np.mean(val_loss_list[-(av.early_stopping + 1):-1]):
            logging.info("Early stopping")
            break
    model.save(sess)
    test_map, test_mean_loss, average_precision_at_k, test_mrr, test_ap, test_rr = \
        evaluate(model, sess, query_features_test, corpus_test, av.num_queries_eval, sparse_walk)
    logging.info(f"average precision at k: \n{average_precision_at_k}\n"
                 f"Average precision: \n{test_ap}\n"
                 f"Reciprocal rank: \n{test_rr}")
    logging.info(f"test mean loss: {test_mean_loss:.5f},"
                 f"\ttest mean average precision: {test_map:.5f},"
                 f"\ttest mean reciprocal rank: {test_mrr:.5f},")
    # plot_loss_map(epoch_list, train_loss_list, train_map_list, val_loss_list, val_map_list, params)


if __name__ == '__main__':
    main()

'''
python train.py --data_path ./data/squad/squad10.pkl --dataset squad --num_queries_train 1000 --num_queries_eval 100 --logfile ./logs/squad_random.log --early_stopping 10 --num_walks 32 --max_length_walk 32 --nlayer1 32 --nlayer2 32 --elayer1 8 --elayer2 8 --nelayer1 32 --nelayer2 32 --walk_method random --sparse_walk False
'''
