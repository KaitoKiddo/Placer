from env import Env
import networkx as nx
from itertools import combinations

def graph_embedding(env, s):

    '''
        parameters:
            nodes: Node_list object from nodes.py
            nets: Net_list object from nets.py
    '''
    
    G = nx.Graph()
    G.add_nodes_from(env.nodes)

    i = 0
    for name in env.nodes:
        G.nodes[name]['x'] = s[i][0]
        G.nodes[name]['y'] = s[i][1]
        i += 1

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