import networkx as nx
import numpy as np
from scipy import spatial
from sklearn.preprocessing import normalize


def restart(restart_prob):
    p = np.random.binomial(1, restart_prob)
    return p


def random_walk(graph, rw_length, transition_prob, start_prob, restart_prob, max_product_graph_nodes):
    nodes_travelled = []
    nodes = list(graph.nodes)
    num_nodes = graph.number_of_nodes()
    curr_node = np.random.choice(np.arange(0, num_nodes), p=start_prob)
    nodes_travelled.append(curr_node)
    next_node = None
    while len(nodes_travelled) <= rw_length:
        if len(nodes_travelled) > 1 and restart(restart_prob):
            # next_node = np.random.choice(nodes_travelled, p=np.ones(len(nodes_travelled)) / len(nodes_travelled))
            next_node = np.random.choice(num_nodes, p=start_prob)
        else:
            if graph.degree[nodes[curr_node]] == 0:
                next_node = np.random.choice(num_nodes, p=start_prob)
            else:
                distribution = transition_prob[curr_node, :]
                next_node = np.random.choice(np.arange(0, num_nodes), p=distribution)
        nodes_travelled.append(next_node)
        curr_node = next_node
    nodes_travelled = list(set(nodes_travelled))
    a = np.zeros((max_product_graph_nodes, max_product_graph_nodes))
    b = np.zeros((max_product_graph_nodes, max_product_graph_nodes))
    a[nodes_travelled, :] = 1.0
    b[:, nodes_travelled] = 1.0
    walk_matrix = np.logical_and(a, b)
    adj = nx.to_numpy_matrix(graph).A
    walk_matrix[:num_nodes, :num_nodes] = np.logical_and(walk_matrix[:num_nodes, :num_nodes], adj)
    return walk_matrix


def sample_random_walks(graph, num_walk, max_length, restart_prob, method, max_product_graph_nodes,
                        word_embedding_dict):
    num_nodes = graph.number_of_nodes()
    walk_matrix = np.zeros((max_length, num_walk, max_product_graph_nodes, max_product_graph_nodes))
    transition_prob = nx.to_numpy_matrix(graph)
    if method == 'random':
        transition_prob = normalize(transition_prob, axis=1, norm='l1')
        start_prob = np.ones(num_nodes) / num_nodes
    elif method == 'biased':
        nodes = list(graph.nodes)
        for i in range(num_nodes):
            for j in range(num_nodes):
                if transition_prob[i, j]:
                    label1, label2 = graph[nodes[i]][nodes[j]]['label']
                    if type(label1) == str:
                        node_label = graph.nodes[nodes[j]]['label']
                        e1 = word_embedding_dict[node_label[0]]
                        e2 = word_embedding_dict[node_label[1]]
                        transition_prob[i, j] = 2 - spatial.distance.cosine(e1, e2)
                        if np.isnan(transition_prob[i, j]):
                            transition_prob[i, j] = 1.0
                    else:
                        transition_prob[i, j] = np.exp(-abs(label1 - label2))

        transition_prob = normalize(transition_prob, axis=1, norm='l1')
        start_prob = np.zeros(num_nodes)
        for i in range(num_nodes):
            label1, label2 = graph.nodes[nodes[i]]['label']
            if type(label1) == str:
                e1 = word_embedding_dict[label1]
                e2 = word_embedding_dict[label2]
                start_prob[i] = 2 - spatial.distance.cosine(e1, e2)
                if np.isnan(start_prob[i]):
                    start_prob[i] = 1.0
            else:
                start_prob[i] = np.exp(-abs(label1 - label2))
        start_prob = start_prob / np.sum(start_prob)
    else:
        raise Exception('invalid walk method given')

    for l in range(0, max_length):
        for k in range(0, num_walk):
            walk_matrix[l, k, :, :] = random_walk(graph, l + 1, transition_prob, start_prob, restart_prob,
                                                  max_product_graph_nodes)

    return walk_matrix


def build_product_graph_and_sample(query_graph, corpus_graphs, num_walk, max_length, restart_prob, method,
                                   max_product_graph_nodes, word_embedding_dict):
    random_walk_sample = []
    if not corpus_graphs:
        return None, None
    for corpus_graph in corpus_graphs:
        graph = nx.tensor_product(query_graph, corpus_graph)
        random_walk_sample.append(
            sample_random_walks(graph, num_walk, max_length, restart_prob, method, max_product_graph_nodes,
                                word_embedding_dict))
    return np.stack(random_walk_sample, axis=0)


def print_rw(nodes_travelled, walk_matrix, graph):
    print("Nodes:")
    nodes = list(graph.nodes)
    num_nodes = graph.number_of_nodes()
    for u in nodes_travelled:
        print(nodes[u], graph.nodes[nodes[u]]['label'])
    print("Edges")
    for i in range(num_nodes):
        for j in range(i, num_nodes):
            if walk_matrix[i, j] == 1.0:
                print(nodes[i], '--', graph[nodes[i]][nodes[j]]['label'], '--', nodes[j])
    return
