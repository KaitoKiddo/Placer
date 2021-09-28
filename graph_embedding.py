from env import Env

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations

def graph_embedding():

    '''
        parameters:
            nodes: Node_list object from nodes.py
            nets: Net_list object from nets.py
    '''
    env = Env()
    G = nx.Graph()
    G.add_nodes_from(env.nodes)
    for net in env.nets:
        G.add_edges_from(list(combinations(net, 2)))

    return G 

if __name__ == '__main__':
    import time
    start = time.time()

    graph_embedding()

    end = time.time()
    print(end - start)