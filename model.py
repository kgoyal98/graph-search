import tensorflow as tf
import logging


class RandomWalkModel(object):
    def __init__(self, num_nodes_query, num_nodes_corpus, dim_node_features, dim_edge_features, node_layer1,
                 node_layer2, edge_layer1, edge_layer2, layer1, layer2, name, delta, dropout, sparse_walk, 
                 walk_shape=None):
        self.query = num_nodes_query
        self.num_nodes_query = self.query
        self.num_nodes_corpus = num_nodes_corpus
        self.dim_node_features = dim_node_features
        self.dim_edge_features = dim_edge_features
        self.node_layer1 = node_layer1
        self.node_layer2 = node_layer2
        self.edge_layer1 = edge_layer1
        self.edge_layer2 = edge_layer2
        self.layer1 = layer1
        self.layer2 = layer2
        self.name = name
        self.delta = delta
        self.walk_shape = walk_shape
        self.dropout = dropout

        self.node_features_query = tf.placeholder(tf.float32, shape=(None, 1, self.dim_node_features),
                                                  name='node_features_query')
        # self.num_nodes_query = tf.shape(self.node_features_query)[0]
        self.node_features_corpus = tf.placeholder(tf.float32, shape=(None, None, 1, self.dim_node_features),
                                                   name='node_features_corpus')
        num_corpus_graphs = tf.shape(self.node_features_corpus)[0]
        # self.num_nodes_corpus = tf.shape(self.node_features_corpus)[1]
        self.num_nodes_product = self.num_nodes_query * self.num_nodes_corpus
        self.node_features_query1 = tf.repeat(self.node_features_query, self.num_nodes_corpus, 0)
        self.node_features_query1 = tf.tile(tf.expand_dims(self.node_features_query1, 0),
                                            (num_corpus_graphs, 1, 1, 1))
        self.node_features_corpus1 = tf.tile(self.node_features_corpus, (1, self.num_nodes_query, 1, 1))
        self.node_features_product = tf.concat([self.node_features_query1, self.node_features_corpus1], 3)

        fc = tf.contrib.layers.fully_connected
        dropout = tf.contrib.layers.dropout
        self.node_features_product = dropout(fc(self.node_features_product, self.node_layer1), self.dropout)
        self.node_features_product = dropout(fc(self.node_features_product, self.node_layer2), self.dropout)  # nc * n1.n2 * 1 * dnp

        self.edge_features_query = tf.placeholder(tf.float32, shape=(None, None, dim_edge_features),
                                                  name='edge_features_query')  # n1 * n1 * de
        self.edge_features_corpus = tf.placeholder(tf.float32, shape=(None, None, None, self.dim_edge_features),
                                                   name='edge_features_corpus')  # nc * n2 * n2 * de

        self.edge_features_query1 = tf.repeat(tf.repeat(self.edge_features_query, self.num_nodes_corpus, 0),
                                              self.num_nodes_corpus, 1)
        self.edge_features_query1 = tf.tile(tf.expand_dims(self.edge_features_query1, 0),
                                            (num_corpus_graphs, 1, 1, 1))
        self.edge_features_corpus1 = tf.tile(self.edge_features_corpus,
                                             (1, self.num_nodes_query, self.num_nodes_query, 1))
        self.edge_features_product = tf.concat([self.edge_features_query1, self.edge_features_corpus1], 3)
        self.edge_features_product = dropout(fc(self.edge_features_product, self.edge_layer1), self.dropout)
        self.edge_features_product = dropout(fc(self.edge_features_product, self.edge_layer2), self.dropout)  # nc * n1.n2 * n1.n2 * deh

        h = tf.concat([tf.tile(self.node_features_product, (1, 1, self.num_nodes_product, 1)),
                       self.edge_features_product,
                       tf.tile(tf.transpose(self.node_features_product, [0, 2, 1, 3]),
                               (1, self.num_nodes_product, 1, 1))], 3)

        h = dropout(fc(h, self.layer1), self.dropout)
        h = dropout(fc(h, self.layer2), self.dropout)
        w = tf.Variable(tf.random.normal([self.layer2, 1], 0, 1, dtype=tf.float32, seed=0))
        g = tf.math.sigmoid
        s = g(tf.tensordot(h, w, [[3], [0]]))
        s = tf.squeeze(s, axis=[-1])  # nc * n1.n2 * n1.n2
        if not sparse_walk:
            self.walks = tf.placeholder(tf.float32, shape=(None, None, None, None, None), name='walks')
            self.max_length_walk = tf.shape(self.walks)[1]
            self.num_walks = tf.shape(self.walks)[2]
            num_edges = tf.reduce_sum(self.walks, [3, 4])
            expanded_s = tf.tile(tf.expand_dims(tf.expand_dims(s, 1), 1), (1, self.max_length_walk, self.num_walks, 1, 1))
            q1 = tf.multiply(self.walks, expanded_s)
            q2 = tf.reduce_sum(q1, [3, 4])
        else:
            self.walks = tf.sparse.placeholder(tf.float32, shape=None, name='walks')
            walks = tf.sparse.reshape(self.walks, shape=self.walk_shape)
            self.max_length_walk = self.walk_shape[1]
            self.num_walks = self.walk_shape[2]
            walks = tf.sparse.reset_shape(walks, self.walk_shape)
            # walks = tf.SparseTensor(values=walks.values, indices=walks.indices, dense_shape=self.walk_shape)  # alternative to above line
            num_edges = tf.sparse.reduce_sum(walks, [3, 4])
            expanded_s = tf.expand_dims(tf.expand_dims(s, 1), 1)
            q1 = walks.__mul__(expanded_s)
            q2 = tf.sparse.reduce_sum(q1, [3, 4])  # nc * max_length_walk * num_walks
        q = q2 / num_edges
        sc = tf.reduce_sum(tf.multiply(q, tf.nn.softmax(q, 2)), 2)
        # sc = tf.reduce_logsumexp(q, axis=2, keepdims=False)
        # self.score = tf.reduce_mean(sc / tf.range(1.0, self.max_length_walk + 1, 1.0), 1)
        self.score = tf.reduce_mean(sc, 1)

        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.vars = {var.name: var for var in variables}
        self.saver = tf.train.Saver(self.vars, max_to_keep=5)

        self.y = tf.placeholder(tf.float32, shape=None, name='y')
        score_diff = tf.tile(self.score, (num_corpus_graphs,)) - tf.repeat(self.score, num_corpus_graphs, 0)
        y_diff = tf.tile(self.y, (num_corpus_graphs,)) - tf.repeat(self.y, num_corpus_graphs, 0)
        self.loss = tf.reduce_sum(tf.nn.relu(self.delta - tf.multiply(score_diff, y_diff)))
        tf.summary.scalar('loss', self.loss)
        self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)

    def save(self, sess):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        save_path = self.saver.save(sess, f"./checkpoints/{self.name}.ckpt")
        logging.info("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        save_path = f"./checkpoints/{self.name}.ckpt"
        self.saver.restore(sess, save_path)
        logging.info("Model restored from file: %s" % save_path)
