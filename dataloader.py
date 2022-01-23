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