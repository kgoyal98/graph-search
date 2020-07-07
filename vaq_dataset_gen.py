import networkx as nx
import numpy as np
import copy
import pickle as pkl
import random
from argparse import ArgumentParser
import os

def print_stats(msg,num_nodes,num_edges,degrees,num_components,node_labels,edge_labels):
    global counter
    counter += 1
    print(msg)
    print("Nodes info: total = ", np.sum(num_nodes), ", average = ", np.sum(num_nodes) / len(num_nodes),
          ", (min,max) =  (", np.min(num_nodes), np.max(num_nodes), ") variance =", np.var(num_nodes))
    print("Edges info: total = ", np.sum(num_edges), ", average = ", np.sum(num_edges) / len(num_edges),
          ", (min,max) =  (", np.min(num_edges), np.max(num_edges), ") variance =", np.var(num_edges))
    print("degrees info: average = ", np.sum(degrees) / len(degrees),
          ", (min,max) =  (", np.min(degrees), np.max(degrees), ") variance =", np.var(degrees))
    print("Label space info : node_label_space ", len(node_labels), " edge_label_space ", len(edge_labels))
    #print("components info: average = ", np.sum(num_components) / len(num_components), ", (min,max) =  (",
    #      np.min(num_components), np.max(num_components), ") variance =", np.var(num_components))
    return

def stats_helper(graphs):
    print("num graphs = ",len(graphs))
    num_edges = []
    num_nodes = []
    degrees = []
    num_components = []
    node_labels = set()
    edge_labels = set()
    iter = 0
    for g in graphs:
        iter += 1
        #print(iter,len(node_labels))
        num_edges.append(g.number_of_edges())
        num_nodes.append(g.number_of_nodes())
        for u in list(g.nodes):
            degrees.append(g.degree[u])
            node_labels.add(g.nodes[u]['label'])
        for e in list(g.edges):
            edge_labels.add(g[e[0]][e[1]]['label'])
        num_components.append(nx.number_connected_components(g))
        #if iter%30000 == 0:
        #     print_stats(iter/30000,num_nodes,num_edges,degrees,num_components,node_labels,edge_labels)
    #print(edge_labels[0:100])
    #print(edge_labels)
    print_stats("finish",num_nodes, num_edges, degrees, num_components,node_labels,edge_labels)
    return


def compute_all_relationships(graph,eps = 0.2):
    directions = {
        "front": [0.754490315914154,-0.6563112735748291,-0.0],
        "below": [-0.0,-0.0,-1.0],
        "behind": [-0.754490315914154,0.6563112735748291,0.0],
        "left": [-0.6563112735748291,-0.7544902563095093,0.0],
        "right": [0.6563112735748291,0.7544902563095093,-0.0],
        "above": [0.0,0.0,1.0]
    }
    name_to_idx = {
        "front":0,
        "behind":1,
        "left":2,
        "right":3,
    }
    all_relationships = {}
    for name, direction_vec in directions.items():
        if name == 'above' or name == 'below': continue
        all_relationships[name] = []
        for i in range(graph.number_of_nodes()):
            coords1 = graph.nodes[i]['pos']
            for j in range(graph.number_of_nodes()):
                if j == i:
                    continue
                coords2 = graph.nodes[j]['pos']
                diff = [coords2[k] - coords1[k] for k in [0, 1, 2]]
                dot = sum(diff[k] * direction_vec[k] for k in [0, 1, 2])
                if dot > eps:
                    if graph.has_edge(i,j):
                        graph[i][j]['label'][name_to_idx[name]] = 1
                    else :
                        graph.add_edge(i,j,label = np.zeros(4))
                        graph[i][j]['label'][name_to_idx[name]] = 1

    return graph

def corpus_gen(params):
    colors ={
        "gray": [87, 87, 87],
        "red": [173, 35, 35],
        "blue": [42, 75, 215],
        "green": [29, 105, 20],
        "brown": [129, 74, 25],
        "purple": [129, 38, 192],
        "cyan": [41, 208, 208],
        "yellow": [255, 238, 51]
    }
    colors_list = ["gray","red","blue","green","brown","purple","cyan","yellow"]
    query_graphs = []
    corpus_graphs = []
    for k in range(0,params['num_queries']):
        q = nx.Graph()
        for u in range(0,params['max_query_nodes']):
            # label will be a feature vector 3 for shape, 1 hot encoded next 3 for colors , next 2 for materials, next 1 for size
            shape_id = np.random.randint(0,3)
            material_id = np.random.randint(6,8)
            size = np.random.randint(0,2)
            size = 0.35*(1+size)
            label_ = np.zeros(9)
            label_[shape_id] = 1
            label_[material_id] = 1
            label_[8] = size
            label_[3:6] = np.asarray(colors[colors_list[np.random.randint(0,len(colors_list))]])/256
            x = np.random.uniform(-3,3)
            y = np.random.uniform(-3,3)
            r = size
            q.add_node(u,label = label_, pos= (x,y,r))
        query_graphs.append(compute_all_relationships(q))
        corpus = []
        for _ in range(params['pos_corpus_per_query']):
            c = copy.deepcopy(q)
            for u in range(0,params['max_corpus_nodes']-params['max_query_nodes']):
                # label will be a feature vector 3 for shape, 1 hot encoded next 3 for colors , next 2 for materials, next 1 for size
                shape_id = np.random.randint(0, 3)
                material_id = np.random.randint(6, 8)
                size = np.random.randint(0, 2)
                size = 0.35*(1 + size)
                label_ = np.zeros(9)
                label_[shape_id] = 1
                label_[material_id] = 1
                label_[8] = size
                label_[3:6] = np.asarray(colors[colors_list[np.random.randint(0, len(colors_list))]]) / 256
                x = np.random.uniform(-3, 3)
                y = np.random.uniform(-3, 3)
                r = size
                c.add_node(u+params['max_query_nodes'],label=label_, pos=(x, y, r))
            c = compute_all_relationships(c)
            corpus.append(c)
        corpus_graphs.append(corpus)
    return query_graphs,corpus_graphs

def add_noise(x,s):
    x+=np.random.normal(0,s)
    x = min(1.0,x)
    x = max(0,x)
    return x

def noisy_corpus(queries,corpus,params):
    s = params['noise']
    for q in queries:
        for u in range(q.number_of_nodes()):
            label_ = q.nodes[u]['label']
            label_[3] = add_noise(label_[3],s)
            label_[4] = add_noise(label_[4],s)
            label_[5] = add_noise(label_[4],s)
            label_[8] = add_noise(label_[8],s)
            q.nodes[u]['label'] = label_

    for corpus_list in corpus:
        for q in corpus_list:
            for u in range(q.number_of_nodes()):
                label_ = q.nodes[u]['label']
                label_[3] = add_noise(label_[3], s)
                label_[4] = add_noise(label_[4], s)
                label_[5] = add_noise(label_[4], s)
                label_[8] = add_noise(label_[8], s)
                q.nodes[u]['label'] = label_

    return queries,corpus

def complete_corpus(params,pos_corpus):
    corpus = []

    for i in range(params['num_queries']):
        corpus_list = pos_corpus[i]
        for j in range(0,params['total_corpus_per_query']-params['pos_corpus_per_query']):
            q = i
            while q == i:
                q = np.random.randint(0,params['num_queries'])
            k = np.random.randint(0,params['pos_corpus_per_query'])
            corpus_list.append(pos_corpus[q][k])
        corpus.append(corpus_list)

    return corpus

def check(query_graphs,train_corpus,params):
    print("Now checking the data")
    #params, , , test_corpus, validation_corpus, train_labels, test_labels, validation_labels = data

    for g in query_graphs:
        if g.number_of_nodes() != params['max_query_nodes']:
            print("Error 1")
        nodes = list(g.nodes)
        #print(nodes)
        if (0 not in nodes) or (1 not in nodes) or (2 not in nodes) or (3 not in nodes) or (4 not in nodes):
            print("Error 2")
        for node in nodes:
            label_ = g.nodes[node]['label']
            if len(label_)!=9:
                print("Error 89")
            for l in label_:
                if l < 0 or l > 1:
                    print("Error 3")
            for node2 in nodes:
                if g.has_edge(node,node2):
                    label_= g.edges[node,node2]['label']
                    if len(label_) != 4:
                        print("Error 79")
                    for l in label_:
                        if l < 0 or l > 1:
                            print("Error 4")
        A = nx.to_numpy_matrix(g)
        for u in nodes:
            for v in nodes:
                if g.has_edge(u,v):
                    A[u,v] = np.sum(g.edges[u,v]['label'])
        if not nx.is_connected(g):
            print("Error disconnected")

    print("query graphs are ok")
    for i in range(len(query_graphs)):
        for g in train_corpus[i]:
            if g.number_of_nodes() != params['max_corpus_nodes']:
                print("Error 1")
            nodes = list(g.nodes)
            for i in range(0,params['max_corpus_nodes']):
                if i not in nodes:
                    print("Error 2")
        #if (0 not in nodes) or (1 not in nodes) or (2 not in nodes) or (3 not in nodes) or (4 not in nodes):

            for node in nodes:
                label_ = g.nodes[node]['label']
                if len(label_) != 9:
                    print("Error 89")
                for l in label_:
                    if l < 0 or l > 1:
                        print("Error 3")
                for node2 in nodes:
                    if g.has_edge(node, node2):
                        label_ = g.edges[node, node2]['label']
                        if len(label_) != 4:
                            print("Error 79")
                        for l in label_:
                            if l < 0 or l > 1:
                                print("Error 4")
            A = nx.to_numpy_matrix(g)
            for u in nodes:
                for v in nodes:
                    if g.has_edge(u, v):
                        A[u, v] = np.sum(g.edges[u, v]['label'])
            if not nx.is_connected(g):
                print("Error disconnected")


def test_corpus_gen(train_queries,params):
    colors = {
        "gray": [87, 87, 87],
        "red": [173, 35, 35],
        "blue": [42, 75, 215],
        "green": [29, 105, 20],
        "brown": [129, 74, 25],
        "purple": [129, 38, 192],
        "cyan": [41, 208, 208],
        "yellow": [255, 238, 51]
    }
    colors_list = ["gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"]
    corpus_graphs = []
    for q in train_queries:
        corpus = []
        for _ in range(params['pos_corpus_per_query']):
            c = copy.deepcopy(q)
            for u in range(0, params['max_corpus_nodes'] - params['max_query_nodes']):
                # label will be a feature vector 3 for shape, 1 hot encoded next 3 for colors , next 2 for materials, next 1 for size
                shape_id = np.random.randint(0, 3)
                material_id = np.random.randint(6, 8)
                size = np.random.randint(0, 2)
                size = 0.35 * (1 + size)
                label_ = np.zeros(9)
                label_[shape_id] = 1
                label_[material_id] = 1
                label_[8] = size
                label_[3:6] = np.asarray(colors[colors_list[np.random.randint(0, len(colors_list))]]) / 256
                x = np.random.uniform(-3, 3)
                y = np.random.uniform(-3, 3)
                r = size
                c.add_node(u + params['max_query_nodes'], label=label_, pos=(x, y, r))
            c = compute_all_relationships(c)
            corpus.append(c)
        corpus_graphs.append(corpus)
    return corpus_graphs

def noisy_data_gen(params):
    train_queries,pos_train_corpus = corpus_gen(params)
    pos_test_corpus = test_corpus_gen(train_queries,params)
    pos_val_corpus = test_corpus_gen(train_queries,params)


    train_corpus = complete_corpus(params,pos_train_corpus)
    test_corpus = complete_corpus(params,pos_test_corpus)
    val_corpus = complete_corpus(params,pos_val_corpus)
    check(train_queries,train_corpus,params)
    check(train_queries, test_corpus, params)
    check(train_queries, val_corpus, params)
    _,pos_test_corpus = noisy_corpus([],val_corpus,params)
    train_queries, train_corpus = noisy_corpus(train_queries, train_corpus, params)
    _, test_corpus = noisy_corpus([], test_corpus, params)
    _, val_corpus = noisy_corpus([], val_corpus, params)

    labels = np.zeros((params['num_queries'],params['total_corpus_per_query']))
    for i in range(params['num_queries']):
        for j in range(0,params['pos_corpus_per_query']):
            labels[i][j] = 1
        for j in range(params['pos_corpus_per_query'],params['total_corpus_per_query']):
            labels[i][j] = 0


    return params,train_queries,train_corpus,test_corpus,val_corpus,labels,labels,labels



def main():
    ap = ArgumentParser()
    ap.add_argument("--data_path", type=str, default="./data/clevr.pkl")
    ap.add_argument("--logfile", type=str, default="./logs/dataset.log")
    ap.add_argument("--num_queries", type=int, default=50)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--seed", type=str, default=0)
    av = ap.parse_args()
    seed = av.seed
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(av.seed)

    params = {
        'max_corpus_nodes': 10,
        'max_query_nodes': 5,
        'num_queries': av.num_queries,
        'pos_corpus_per_query': 20,
        'total_corpus_per_query': 100,
        'noise': av.noise
    }
    data = noisy_data_gen(params)
    outfile = open(av.data_path, 'wb')
    pkl.dump(data, outfile)
    outfile.close()


main()