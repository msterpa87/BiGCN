import pandas as pd
import numpy as np
from os import listdir
from networkx.convert_matrix import to_scipy_sparse_matrix
from collections import defaultdict
from spektral.data import Dataset
from spektral.data.graph import Graph
from contextlib import suppress
from spektral.utils import one_hot
from utils import *
from os.path import join as joinpath
from scipy.sparse import load_npz

################################################################
#                                                              #
#                         WICO dataset                         #
#                                                              #
################################################################
class WICO(Dataset):
    def __init__(self, path="./dataset/WICO/", root_edges=True,
                 time_delay_edges=True, hours=1):
        self.custom_path = path
        self.root_edges = root_edges
        self.time_delay_edges = time_delay_edges
        self.hours = hours
        self.labels = {x: i for i, x in enumerate(listdir(self.custom_path))}

        super().__init__()

    def read(self):
        user_info, user_tweet_count = load_users_info(self.custom_path)

        graph_spektral_list = []

        for y,graph_type in enumerate(listdir(self.custom_path)):
            pathname = f"{self.custom_path}/{graph_type}/"
            subgraphs_list = list(map(int, filter(str.isnumeric, listdir(pathname))))

            for graph_id in sorted(subgraphs_list):
                G = load_graph_from_file(f"{pathname}/{graph_id}",
                                        root_edges=True, time_delay_edges=True)
                
                if G is None:
                    continue

                A = to_scipy_sparse_matrix(G, dtype=float)

                # build node features vectors
                df = pd.read_csv(f"{pathname}/{graph_id}/nodes.csv")

                # features are (followers, friends, number of tweets, tweet_timestamp)
                x = np.zeros((len(G.nodes), 4), dtype=float)

                # keep only nodes in the dataframe that have are in the graph
                tuples = [x[1:] for x in df.itertuples() if x[1] in G.nodes]

                for i, (user_id, tweet_time, friends, followers) in enumerate(tuples):
                    friends, followers = (0, 0)

                    with suppress(KeyError):
                        friends, followers = user_info[user_id]
                    
                    n_tweets = user_tweet_count[user_id]

                    x[i] = np.array([friends, followers, n_tweets, tweet_time], dtype=float)

                graph_spektral_list.append(Graph(a=A, x=x, y=one_hot(y, 3)))
        
        return graph_spektral_list


################################################################
#                                                              #
#         FakeNewsNet dataset (politifact and gossipcop)       #
#                                                              #
################################################################
class FakenNewsNet(Dataset):
    def __init__(self, path):
        self.custom_path = path
        super().__init__()
    
    def read(self):
        graph_spektral_list = []

        for y, label in enumerate(["real", "fake"]):
            subgraphs_path = f"{self.custom_path}/{label}/subgraphs"
            features_path = f"{self.custom_path}/{label}/features"

            for filename in listdir(subgraphs_path):
                # load news subgraphs
                edge_list = load_edge_list(f"{subgraphs_path}/{filename}", sep=", ")
                G = nx.from_edgelist(edge_list, create_using=nx.DiGraph)

                A = to_scipy_sparse_matrix(G, dtype=float)

                # load node features
                features = load_features(f"{features_path}/{filename}")

                # special node has all features = 0
                features[0] = np.zeros(8)
                x = np.zeros((len(G.nodes), 8), dtype=float)

                # create feature matrix respecting graph nodes ordering
                for i,user_id in enumerate(G.nodes):
                    x[i] = features[user_id]
                
                graph_spektral_list.append(Graph(a=A, x=x, y=y))
        
        return graph_spektral_list

class PolitiFact(Dataset):
    def __init__(self):
        self.custom_path = "./dataset/politifact/"
        super().__init__()
    
    def read(self):
        graph_spektral_list = []
        path = self.custom_path

        A = load_edge_list(joinpath(path, "A.txt"), sep=", ")
        node_graph_id = np.load(joinpath(path, "node_graph_id.npy"))

        # split adjacency matrix in subgraphs
        graph_list = defaultdict(list)

        # load graph labels (0 = real, 1 = fake)
        labels = np.load(joinpath(path, "graph_labels.npy"))

        for edge in A:
            idx = node_graph_id[edge[0]]    # graph ID of edge
            graph_list[idx].append(edge)
        
        # load node features
        profile = load_npz(joinpath(path, "new_profile_feature.npz")).todense()
        bert = load_npz(joinpath(path, "new_bert_feature.npz")).todense()
        spacey = load_npz(joinpath(path, "new_spacy_feature.npz")).todense()
        content = load_npz(joinpath(path, "new_content_feature.npz")).todense()

        features_size = profile.shape[1] + bert.shape[1] + spacey.shape[1] + content.shape[1]

        for graph_id, edge_list in graph_list.items():
            # create sparse adjacency matrix
            G = nx.from_edgelist(edge_list, create_using=nx.DiGraph)
            A = to_scipy_sparse_matrix(G, dtype=float)

            # create node features
            x = np.zeros((len(G.nodes), features_size), dtype=float)

            for i, user_id in enumerate(G.nodes):
                x[i] = np.concatenate((profile[user_id], bert[user_id],
                                      spacey[user_id], content[user_id]), axis=1)
            
            # graph label (fake/real)
            y = labels[graph_id]
            
            graph_spektral_list.append(Graph(a=A, x=x, y=y))
        
        return graph_spektral_list

