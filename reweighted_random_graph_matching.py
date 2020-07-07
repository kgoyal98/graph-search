import numpy as np
from utils import average_precision, precision_at_k_list, reciprocal_rank
import pickle as pkl
from argparse import ArgumentParser


def reweighted_random_graph_matching(weight_matrix, num_nodes, reweight_factor=0.2, inflation_factor=30, ):
  W = weight_matrix
  alpha = reweight_factor
  beta = inflation_factor
  d_max = max(np.sum(W, axis=1))
  P = np.true_divide(W, d_max)
  x_next = np.ones(W.shape[0]) / W.shape[0]
  x_curr = np.ones(W.shape[0])
  while abs(np.sum(x_next - x_curr)) > 1e-6:
    x_curr = x_next
    x_next = np.matmul(x_next, P)
    # find y
    y_next = np.exp(beta * x_next / np.max(x_next))
    y_next = np.reshape(y_next, (num_nodes[0], num_nodes[1]))
    y_curr = np.ones((num_nodes[0], num_nodes[1]))
    while abs(np.sum(np.sum(y_next - y_curr))) > 1e-6:
      y_curr = y_next
      y_next = np.transpose(np.transpose(y_next) / np.sum(y_next, axis=1))
      y_next = y_next / np.sum(y_next, axis=0)
    x_next = alpha * x_next + (1 - alpha) * np.reshape(y_next, x_next.shape)
    x_next = x_next / np.sum(x_next)

  X = x_next
  return np.matmul(np.matmul(X, W), np.transpose(X))


def get_score(query_graph, corpus_graph):
  nq = query_graph.number_of_nodes()
  nc = corpus_graph.number_of_nodes()
  W = np.zeros((nq * nc, nq * nc))
  s = 0.15
  nodes_c = list(corpus_graph.nodes)
  nodes_q = list(query_graph.nodes)
  for i in range(nq):
    for a in range(nc):
      for j in range(nq):
        for b in range(nc):
          if query_graph.has_edge(nodes_q[i], nodes_q[j]) and corpus_graph.has_edge(nodes_c[a], nodes_c[b]):
            # print(nodes_q[i],nodes_q[j])
            a0 = query_graph.nodes[nodes_q[i]]['label']
            a1 = corpus_graph.nodes[nodes_c[a]]['label']
            b0 = query_graph.edges[nodes_q[i], nodes_q[j]]['label']
            b1 = corpus_graph.edges[nodes_c[a], nodes_c[b]]['label']
            c0 = query_graph.nodes[nodes_q[j]]['label']
            c1 = corpus_graph.nodes[nodes_c[b]]['label']
            W[i * nq + a, j * nq + b] = np.exp(-( (a0-a1)**2+(b0-b1)**2+(c0-c1)**2) / s)

  return reweighted_random_graph_matching(W, (nq, nc))

def check(train_corpus):
  for i in range(len(train_corpus)):
    for c in train_corpus[i]:
      for e in c.edges():
        u,v = e
        if not c.has_edge(u,v) or not c.has_edge(v,u):
          print("Oh shit")
          exit()

def main():
  ap = ArgumentParser()
  ap.add_argument("--data_path", type=str, default="./data2.pkl")
  ap.add_argument("--num_queries", type=int, default=40)
  av = ap.parse_args()
  pickle_in = open(av.data_path, "rb")
  data_params, query_graphs, train_corpus, test_corpus, validation_corpus, train_labels, test_labels, validation_labels = \
    pkl.load(pickle_in)
  pickle_in.close()
  check(train_corpus)
  print("checked")
  maps = []
  average_precision_at_k_list = []
  reciprocal_rank_list = []
  for j in range(av.num_queries):
    print(j)
    graph = query_graphs[j]
    scores = []
    for i in range(len(test_corpus[j])):
      corpus_graph = test_corpus[j][i]
      s = get_score(graph, corpus_graph)
      scores.append(s)
    map = average_precision(validation_labels[j, :], scores)
    average_precision_at_k_list.append(precision_at_k_list(validation_labels[j, :], scores))
    reciprocal_rank_list.append(reciprocal_rank(validation_labels[j, :], scores))
    maps.append(map)
  average_precision_at_k = np.average(average_precision_at_k_list, axis=0)
  mrr = np.mean(reciprocal_rank_list)
  print(f"average precision at k: \n{average_precision_at_k}")
  print(f"\t\tmean average precision:\t{np.mean(maps)}"
        f"\t\tmean reciprocal rank:\t{mrr}")


if __name__ == '__main__':
  main()


