import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.activations import relu
from spektral.layers import GraphSageConv
from spektral.data.graph import Graph
from contextlib import suppress
import spektral
import numpy as np
from spektral.data.graph import Graph
from spektral.layers import DiffPool

class GNN(tf.keras.Model):
    def __init__(self, hidden_channels, out_channels, add_linear=True):
        super(GNN, self).__init__()
        self.conv1 = GraphSageConv(hidden_channels)
        self.bn1 = BatchNormalization()
        self.conv2 = GraphSageConv(hidden_channels)
        self.bn2 = BatchNormalization()
        self.conv3 = GraphSageConv(out_channels)
        self.bn3 = BatchNormalization()
        
        if add_linear:
            self.lin = Dense(out_channels)
        else:
            self.lin = None
    
    def call(self, inputs):
        x, adj = inputs
        
        x1 = self.bn1(relu(self.conv1(inputs)))
        x2 = self.bn2(relu(self.conv2([x1, adj])))
        x3 = self.bn3(relu(self.conv3([x2, adj])))

        x = tf.concat([x1, x2, x3], axis=-1)

        if self.lin is not None:
            x = relu(self.lin(x))

        return x

def dense_diff_pool(x, adj, s):
    # https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/dense/diff_pool.html#dense_diff_pool
    # add batch dimension if necessary
    with suppress(TypeError):
        adj = tf.sparse.to_dense(adj)
        s = tf.sparse.to_dense(adj)

    #x = tf.expand_dims(x, axis=0) if len(x.shape) == 2 else x
    #adj = tf.expand_dims(adj, axis=0) if len(adj.shape) == 2 else adj
    #s = tf.expand_dims(s, axis=0) if len(s.shape) == 2 else s

    #batch_size, num_nodes, _ = x.shape  # used when maks is implemented

    s = tf.nn.softmax(s, axis=-1)
    st = tf.transpose(s, (1, 0))

    out = tf.matmul(st, x)
    out_adj = tf.matmul(tf.matmul(st, adj), s)

    link_loss = adj - tf.matmul(s, st)
    link_loss = tf.norm(link_loss, ord=2)
    link_loss = link_loss / tf.size(adj, out_type=tf.dtypes.float32)

    ent_loss = tf.reduce_mean(tf.reduce_sum(-s * tf.math.log(s + 1e-15), axis=-1))

    return out, tf.sparse.from_dense(out_adj), link_loss, ent_loss

class Net(tf.keras.Model):
    def __init__(self, num_classes=6, max_nodes=200):
        super(Net, self).__init__()

        num_nodes = np.ceil(0.25 * max_nodes).astype(int)
        self.gnn1_pool = GNN(64, num_nodes)
        self.gnn1_embed = GNN(64, 64, add_linear=False)

        num_nodes = np.ceil(0.25 * num_nodes).astype(int)
        self.gnn2_pool = GNN(64, num_nodes)
        self.gnn2_embed = GNN(64, 64, add_linear=False)

        self.gnn3_embed = GNN(64, 64, add_linear=False)

        self.lin1 = Dense(64, activation='relu')
        self.lin2 = Dense(num_classes, activation='sigmoid')
    
    def call(self, inputs):
        x, adj = inputs
        s = self.gnn1_pool(inputs)
        x = self.gnn1_embed([x, adj])

        x, adj, l1, e1 = dense_diff_pool(x, adj, s)

        s = self.gnn2_pool([x, adj])
        x = self.gnn2_embed([x, adj])

        x, adj, l2, e2 = dense_diff_pool(x, adj, s)

        x = self.gnn3_embed([x, adj])

        x = tf.reduce_mean(x, axis=1)

        if len(x.shape) == 1:
            x = tf.expand_dims(x, axis=0)

        x = self.lin1(x)
        x = self.lin2(x)

        return x, l1 + l2, e1 + e2

class GNNCL(tf.keras.Model):
    def __init__(self, num_classes=6, max_nodes=200, channels=64, shrinkage=0.25):
        super(GNNCL, self).__init__()

        num_nodes = np.ceil(shrinkage * max_nodes).astype(int)
        self.pool1 = DiffPool(num_nodes, channels=channels)

        num_nodes = np.ceil(shrinkage * num_nodes).astype(int)
        self.pool2 = DiffPool(num_nodes, channels=channels)

        self.gnn_embed = GNN(channels, channels, add_linear=False)

        self.lin1 = Dense(channels, activation='relu')
        self.lin2 = Dense(num_classes, activation='sigmoid')
    
    def call(self, inputs):
        x, adj = inputs
        x, adj = self.pool1([x, adj])
        adj = tf.sparse.from_dense(adj)

        x, adj = self.pool2([x, adj])
        adj = tf.sparse.from_dense(adj)
        
        x = self.gnn_embed([x, adj])
        x = tf.reduce_mean(x, axis=1)

        if len(x.shape) == 1:
            x = tf.expand_dims(x, axis=0)

        x = self.lin1(x)
        x = self.lin2(x)

        return x