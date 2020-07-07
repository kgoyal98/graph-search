import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import spacy

nlp = spacy.load("en_core_web_md")


def shuffle(l1, l2):
    import random
    x = list(zip(l1, l2))
    random.shuffle(x)
    l3, l4 = zip(*x)
    return list(l3), list(l4)


def reciprocal_rank(labels, predictions):
    labels, predictions = shuffle(labels, predictions)
    sorted_predictions = sorted(((e, i) for i, e in enumerate(predictions)), reverse=True)
    rr = 0
    for i in range(len(sorted_predictions)):
        if labels[sorted_predictions[i][1]] == 1.0:
            rr = 1.0 / (i + 1)
            break
    return rr


def precision_at_k(labels, predictions, k):
    sorted_predictions = sorted(((e, i) for i, e in enumerate(predictions)), reverse=True)
    assert (k <= len(labels))
    tp = 0
    for i in range(k):
        if labels[sorted_predictions[i][1]] == 1.0:
            tp += 1

    return 1.0 * tp / k


def precision_at_k_list(labels, predictions):
    list = []
    for k in range(len(labels)):
        list.append((precision_at_k(labels, predictions, k + 1)))
    return list


def average_precision(labels, predictions):
    labels, predictions = shuffle(labels, predictions)
    sorted_predictions = sorted(((e, i) for i, e in enumerate(predictions)), reverse=True)
    precision = []
    j = 1
    for i in range(len(sorted_predictions)):
        if labels[sorted_predictions[i][1]] == 1.0:
            precision.append(1.0 * j / (i + 1))
            j += 1
    return np.average(precision)


def random_average_precision(labels):
    n = len(labels)
    predictions = np.arange(n)
    ap = []
    for _ in range(1000):
        np.random.shuffle(predictions)
        ap.append(average_precision(labels, predictions))

    return np.mean(ap)


def expected_average_precision(labels):
    # Exact Expected Average Precision of the Random Baseline for System Evaluation
    # https://ufal.mff.cuni.cz/pbml/103/art-bestgen.pdf

    def dhyper(i, R, N, n):
        from scipy.special import comb
        return comb(R, i) * comb(N-R, n-i) / comb(N, n)

    ap = 0
    N = len(labels)
    R = int(np.sum(labels))
    for i in range(1, R+1):
        for n in range(i, N-R+i+1):
            ap += dhyper(i, R, N, n) * (i/n) * (i/n)
    ap /= R
    return ap


def random_reciprocal_rank(labels):
    n = len(labels)
    predictions = np.arange(n)
    rr = []
    for _ in range(1000):
        np.random.shuffle(predictions)
        rr.append(reciprocal_rank(labels, predictions))

    return np.mean(rr)


def plot(graph):
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, labels=nx.get_node_attributes(graph, 'label'))
    edge_labels = nx.get_edge_attributes(graph, 'label')
    nx.draw_networkx_edge_labels(graph, pos, labels=edge_labels)
    plt.show()
    # plt.savefig("./plots/temp.png")


def train(model, sess, query_features, corpus, num_queries, sparse_walk):
    losses = []
    maps = []
    average_precision_at_k_list = []
    reciprocal_rank_list = []
    num_corpus = corpus[0][0].shape[0]
    for i in range(num_queries):
        feed_dict = {model.node_features_query: query_features[i][0],
                     model.node_features_corpus: corpus[0][i],
                     model.edge_features_query: query_features[i][1],
                     model.edge_features_corpus: np.reshape(corpus[1][i].todense().A, corpus[4]),
                     model.walks: sparse_to_tuple(corpus[2][i]) if sparse_walk else np.reshape(corpus[2][i].todense().A, corpus[5]),
                     model.y: corpus[3][i, :]}
        [_, scores, loss] = sess.run([model.optimizer, model.score, model.loss], feed_dict=feed_dict)
        losses.append(loss)
        map = average_precision(corpus[3][i, :], scores)
        maps.append(map)
        average_precision_at_k_list.append(precision_at_k_list(corpus[3][i, :], scores))
        reciprocal_rank_list.append(reciprocal_rank(corpus[3][i, :], scores))
    mean_map = np.mean(maps)
    mean_loss = np.mean(losses)
    average_precision_at_k = np.average(average_precision_at_k_list, axis=0)  # list
    mrr = np.mean(reciprocal_rank_list)
    return mean_map, mean_loss, average_precision_at_k, mrr


def evaluate(model, sess, query_features, corpus, num_queries, sparse_walk):
    losses = []
    maps = []
    average_precision_at_k_list = []
    reciprocal_rank_list = []
    num_corpus = corpus[0][0].shape[0]
    for i in range(num_queries):
        feed_dict = {model.node_features_query: query_features[i][0],
                     model.node_features_corpus: corpus[0][i],
                     model.edge_features_query: query_features[i][1],
                     model.edge_features_corpus: np.reshape(corpus[1][i].todense().A, corpus[4]),
                     model.walks: sparse_to_tuple(corpus[2][i]) if sparse_walk else np.reshape(corpus[2][i].todense().A, corpus[5]),
                     model.y: corpus[3][i, :]}
        [scores, loss] = sess.run([model.score, model.loss], feed_dict=feed_dict)
        losses.append(loss)
        map = average_precision(corpus[3][i, :], scores)
        maps.append(map)
        average_precision_at_k_list.append(precision_at_k_list(corpus[3][i, :], scores))
        reciprocal_rank_list.append(reciprocal_rank(corpus[3][i, :], scores))
    mean_map = np.mean(maps)
    mean_loss = np.mean(losses)
    average_precision_at_k = np.average(average_precision_at_k_list, axis=0)  # list
    mrr = np.mean(reciprocal_rank_list)
    return mean_map, mean_loss, average_precision_at_k, mrr, maps, reciprocal_rank_list


def plot_loss_map(epoch_list, train_loss_list, train_map_list, val_loss_list, val_map_list, params):
    fig = plt.figure()
    fig, axs = plt.subplots(1, 2)
    fig.suptitle(params)
    axs[0].clear()
    axs[0].plot(epoch_list, train_loss_list, 'r', label='training loss')
    axs[0].plot(epoch_list, val_loss_list, 'b', label='validation loss')
    axs[0].set_xlabel('epochs')
    axs[0].set_title('loss')
    axs[0].legend()

    axs[1].clear()
    axs[1].plot(epoch_list, train_map_list, 'r', label='training map')
    axs[1].plot(epoch_list, val_map_list, 'b', label='validation map')
    axs[1].set_xlabel('epochs')
    axs[1].set_title('mean average precision')
    axs[1].legend()
    fig.savefig('./plots/' + params + '.png')
    # fig.canvas.draw()


def param_dict_to_str(d):
    s = ""
    for k, v in d.items():
        s += str(k) + "=" + str(v) + "-"
    return s


def indices_to_one_hot(indices, dim):
    return np.eye(dim)[indices]


word_found = 0
word_not_found = 0


def word_embedding(word, type=None):
    global nlp, word_found, word_not_found
    tokens = nlp(word)
    for token in tokens:
        if token.has_vector:
            word_found += 1
        else:
            word_not_found += 1
        return token.vector


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    from scipy import sparse

    def to_tuple(mx):
        if not sparse.isspmatrix_coo(mx):
            mx = mx.tocoo()
        indices = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return indices, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx
