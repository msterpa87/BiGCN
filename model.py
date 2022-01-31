from turtle import forward
from libcst import Global
import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Flatten
from tensorflow.keras.activations import relu
from spektral.layers import GraphSageConv, GCNConv, GCSConv, GATConv, DiffPool, GlobalAvgPool, GlobalMaxPool
from spektral.data.graph import Graph
from contextlib import suppress
import spektral
import numpy as np
from spektral.data.graph import Graph

class GNN(tf.keras.Model):
    def __init__(self, hidden_channels, out_channels, add_linear=True):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(hidden_channels, activation='relu')
        #self.bn1 = BatchNormalization()
        self.conv2 = GCNConv(hidden_channels, activation='relu')
        #self.bn2 = BatchNormalization()
        self.conv3 = GCNConv(out_channels, activation='relu')
        #self.bn3 = BatchNormalization()
        
        if add_linear:
            self.lin = Dense(out_channels, activation='relu')
        else:
            self.lin = None
    
    def call(self, inputs):
        x, adj = inputs
        
        x1 = self.bn1(self.conv1(inputs))
        x2 = self.bn2(self.conv2([x1, adj]))
        x3 = self.bn3(self.conv3([x2, adj]))

        x = tf.concat([x1, x2, x3], axis=-1)

        if self.lin is not None:
            x = self.lin(x)

        return x

class GNNCL(tf.keras.Model):
    def __init__(self, num_classes=1, max_nodes=500, channels=64, shrinkage=0.25, batch_size=32):
        super(GNNCL, self).__init__()

        num_nodes = max(np.ceil(shrinkage * max_nodes).astype(int), batch_size)
        self.pool1 = DiffPool(channels, channels=channels, activation='relu')

        num_nodes = max(np.ceil(shrinkage * num_nodes).astype(int), batch_size)
        self.pool2 = DiffPool(channels, channels=channels, activation='relu')

        self.gnn_embed = GNN(channels, channels, add_linear=False)

        self.lin1 = Dense(channels, activation='relu')
        self.lin2 = Dense(num_classes, activation='sigmoid')
    
    def call(self, inputs):
        x, adj = inputs
        x, adj = self.pool1([x, adj])
        x, adj = self.pool2([x, adj])
        adj = tf.sparse.from_dense(adj)

        x = self.gnn_embed([x, adj])

        x = tf.reduce_mean(x, axis=2)
        
        x = self.lin1(x)
        x = self.lin2(x)

        return tf.reshape(x, (-1,1))

class GCNFN(tf.keras.Model):
    def __init__(self, num_classes=1, channels=64):
        super(GCNFN, self).__init__()

        self.conv1 = GATConv(channels * 2)
        self.conv2 = GATConv(channels * 2)

        self.fc1 = Dense(channels, activation="selu")
        
        self.fc2 = Dense(num_classes)

        self.pool = GlobalAvgPool()
        self.dropout = Dropout(0.5)
        
    
    def call(self, inputs):
        x, adj = inputs
    
        x = self.conv1([x, adj])
        x = self.conv2([x, adj])
        x = self.pool(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return tf.nn.log_softmax(x, axis=-1)

class FirstNet(tf.keras.Model):
    def __init__(self) -> None:
        super(FirstNet, self).__init__()
        self.conv1 = GCNConv(16, activation='relu')
        self.conv2 = GCNConv(32, activation='relu')
        self.conv3 = GCNConv(64, activation='relu')
        self.conv4 = GCNConv(1, activation='relu')
        self.dropout = 0.1
        self.pool = GlobalMaxPool()
    
    def call(self, inputs):
        x, adj = inputs

        x = self.conv1([x, adj])
        x = Dropout(self.dropout)(x)
        x = self.conv2([x, adj])
        x = Dropout(self.dropout)(x)
        x = self.conv3([x, adj])
        x = Dropout(self.dropout)(x)
        x = self.conv4([x, adj])
        x = self.pool(x)
        x = tf.nn.log_softmax(x, axis=1)

        return x
