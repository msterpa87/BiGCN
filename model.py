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

class GNNCL(tf.keras.Model):
    def __init__(self, num_classes=1, max_nodes=200, channels=64, shrinkage=0.25, batch_size=32):
        super(GNNCL, self).__init__()

        #num_nodes = max(np.ceil(shrinkage * max_nodes).astype(int), batch_size)
        self.pool1 = DiffPool(channels, channels=channels)

        #num_nodes = max(np.ceil(shrinkage * num_nodes).astype(int), batch_size)
        self.pool2 = DiffPool(channels, channels=channels)

        self.gnn_embed = GNN(channels, channels, add_linear=False)

        self.lin1 = Dense(channels, activation='relu')
        self.lin2 = Dense(num_classes, activation='softmax')
    
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