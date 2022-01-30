import json
from os import listdir, mkdir
from os.path import isfile
import networkx as nx
from contextlib import suppress
from tqdm import tqdm
from datetime import datetime

MAX_TIME_DIFF = 1
MIN_SUBGRAPH_EDGES = 5

def load_twitter_credentials(filename="twitter_api.txt"):
    with open(filename) as f:
        lines = list(map(str.strip, f.readlines()))
    return dict(list(map(lambda x: x.replace("\"", "").split(' = '), lines)))

def str_to_time(timestamp):
    format = "%a %b %d %H:%M:%S %z %Y"
    return datetime.strptime(timestamp, format).replace(tzinfo=None)

def months_from_creation(date):
    twitter_creation = datetime.strptime("Mar 1 2006", "%b %d %Y")
    return (date.year - twitter_creation.year)*12 + date.month - twitter_creation.month

def save_edge_list(edge_list, pathname):
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

def load_edge_list(path):
    with open(path) as f:
        lines = list(map(str.strip, f.readlines()))
    
    edge_list = []

    for line in lines:
        u, v = list(map(int, line.split(', ')))
        edge_list.append((u,v))
    
    return edge_list

class TwitterNode(object):
    def __init__(self, pathname) -> None:
        with open(pathname) as f:
            node = json.load(f)
        
        user = node['user']
        self.followers_count = user['followers_count']
        self.friends_count = user['friends_count']
        self.statuses_count = user['statuses_count']
        self.favourites_count = user['favourites_count']
        self.lists_count = user['listed_count']
        self.verified = int(user['verified'])
        self.user_created_at = months_from_creation(str_to_time(user['created_at']))
        self.user_id = user['id']
        self.tweet_id = node['id']
        self.mentions = [x['id'] for x in node['entities']['user_mentions']]
        self.created_at = str_to_time(node['created_at'])
    
    def get_features_vector(self):
        return [self.verified, self.user_created_at, self.followers_count,
                self.friends_count, self.lists_count, self.favourites_count,
                self.statuses_count, self.created_at]