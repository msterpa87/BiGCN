from collections import defaultdict
import networkx as nx
import pandas as pd
from os import listdir

def load_edge_list(filename):
    edges = []

    with open(filename, "r") as f:
        for line in f:
            u,v = line.split()
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
    edge_list = load_edge_list(f"{pathname}/edges.txt")

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