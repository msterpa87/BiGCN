from collections import defaultdict
import networkx as nx
import pandas as pd
from os import listdir
import json
from datetime import datetime
import tweepy
import numpy as np
from spektral.data import DisjointLoader, BatchLoader

MAX_TIME_DIFF = 10      # max number of hours to add edge between tweets
MIN_SUBGRAPH_EDGES = 5  # min number of edges to create a news subgraph

def load_edge_list(filename, sep=""):
    edges = []

    with open(filename, "r") as f:
        for line in f:
            u,v = line.split(sep)
            edges.append((int(u), int(v)))
    
    return edges

def load_graph_from_file(pathname, root_edges=True, time_delay_edges=True, hours=1,
                         create_using=None):
    """[summary]

    Args:
        pathname ([type]): [description]
        root_edges (bool, optional): Adds a node to the root for each vertex. Defaults to True.
        time_delay_edges (bool, optional): Adds edge to root tweet for tweets made
                                           within hours from it. Defaults to True.
        create_using (nx.Graph, optional): Defines the type of Graph to use.
                                           Defaults to None (undirected graph)

    Returns:
        [type]: [description]
    """
    edge_list = load_edge_list(f"{pathname}/edges.txt", sep = " ")

    if(len(edge_list) == 0):
        return None

    G = nx.from_edgelist(edge_list, create_using=create_using)

    df = pd.read_csv(f"{pathname}/nodes.csv")

    if root_edges:
        # add edges from root tweet to all other nodes
        root_id = df.sort_values("time").iloc[0,0]
        new_edges = [(root_id, x) for x in G if x != root_id]
        G.add_edges_from(new_edges)
    
    if time_delay_edges:
        # add edges from root to tweet < 10 hours from root tweet
        df["time"] = (df["time"] / 3600).astype(float)
        user_ids = df[df["time"] < hours]["id"].tolist()
        new_edges = [(root_id, x) for x in user_ids]

    return G

def load_users_info(path):
    """[summary]

    Returns:
        [type]: [description]
    """
    user_tweet_count = defaultdict(int)
    user_info = defaultdict()

    for graph_type in listdir(path):
        pathname = f"{path}/{graph_type}/"
        subgraphs_list = list(map(int, filter(str.isnumeric, listdir(pathname))))

        for graph_id in sorted(subgraphs_list)[:50]:
            df = pd.read_csv(f"{pathname}/{graph_id}/nodes.csv")
            
            for _, user_id, _, friends, followers in df.itertuples():
                user_tweet_count[user_id] += 1
                user_info[user_id] = (friends, followers)

    return user_info, user_tweet_count

def get_twitter_api(filename="twitter_api.txt"):
    with open(filename) as f:
        lines = list(map(str.strip, f.readlines()))
    keys = dict(list(map(lambda x: x.replace("\"", "").split(' = '), lines)))
    auth = tweepy.OAuthHandler(keys["consumer_key"], keys["consumer_secret"])
  
    # set access to user's access key and access secret 
    auth.set_access_token(keys["access_token"], keys["access_token_secret"])

    # calling the api 
    api = tweepy.API(auth, wait_on_rate_limit=True, retry_errors=[429], retry_delay=100)

    return api

def str_to_time(timestamp):
    format = "%a %b %d %H:%M:%S %z %Y"
    return datetime.strptime(timestamp, format).replace(tzinfo=None)

def months_from_creation(date):
    twitter_creation = datetime.strptime("Mar 1 2006", "%b %d %Y")
    return (date.year - twitter_creation.year)*12 + date.month - twitter_creation.month

def save_edge_list(edge_list, pathname):
    edge_list = list(set(edge_list))

    with open(pathname, "w") as f:
        for (u,v) in edge_list:
            f.write(f"{u}, {v}\n")

def save_node_features(node_features, pathname):
    with open(pathname, "w") as f:
        for user_id, vector in node_features.items():
            feature_str = ", ".join(list(map(str, vector)))
            f.write(f"{user_id}, {feature_str}\n")

def tweet_hours_diff(x, y):
    return round((x.created_at - y.created_at).total_seconds() / 3600, 2)

def load_features(path):
    with open(path) as f:
        lines = list(map(str.strip, f.readlines()))
    
    features = {}

    for line in lines:
        fields = list(map(int, line.split(', ')))
        features[fields[0]] = fields[1:]
    
    return features

class TwitterNode(object):
    def __init__(self, pathname) -> None:
        with open(pathname) as f:
            node = json.load(f)
        
        user = node['user']
        self.followers_count = user['followers_count']
        self.friends_count = user['friends_count']
        self.statuses_count = user['statuses_count']
        self.favourites_count = user['favourites_count']
        self.retweeted = node['retweeted_status']
        self.lists_count = user['listed_count']
        self.verified = int(user['verified'])
        self.user_created_at = months_from_creation(str_to_time(user['created_at']))
        self.user_id = user['id']
        self.tweet_id = node['id']
        self.mentions = [x['id'] for x in node['entities']['user_mentions']]
        self.created_at = str_to_time(node['created_at'])

        self.retweeted_from = None
        if 'reweeted_from' in node.keys():
            self.retweeted_from = node['retweeted_from']
    
    def get_features_vector(self):
        return [self.verified, self.user_created_at, self.followers_count,
                self.friends_count, self.lists_count, self.favourites_count,
                self.statuses_count, self.created_at]

def random_split(data, train_pct=0.75, train_epochs=5, batch_size=1, seed=None):
    """ returns two DataLoaders, resp. for train and test set 
        according to the input split percentage """
    if seed is not None:
        np.random.default_rng(seed)

    np.random.shuffle(data)
    n = len(data)
    train_size = int(n*train_pct)

    train_set = data[:train_size]
    test_set = data[train_size:]

    train_loader = BatchLoader(train_set, batch_size=batch_size, epochs=train_epochs, shuffle=True)
    test_loader = BatchLoader(test_set, batch_size=batch_size, epochs=1)

    return train_loader, test_loader