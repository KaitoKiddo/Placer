from env import Env
import networkx as nx
from itertools import combinations

def graph_embedding(env):

    '''
        parameters:
            nodes: Node_list object from nodes.py
            nets: Net_list object from nets.py
    '''
    
    G = nx.Graph()
    G.add_nodes_from(env.nodes)
    for net in env.nets:
        G.add_edges_from(list(combinations(net, 2)))

    return G 

if __name__ == '__main__':
    import time
    start = time.time()
    env = Env()
    graph_embedding(env)

    end = time.time()
    print(end - start)