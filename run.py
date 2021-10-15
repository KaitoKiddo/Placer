from networkx.algorithms import dominance
from graph_embedding import graph_embedding
from env import Env

env = Env()

s = env.reset()
G = graph_embedding(env, s)
r = 0
for i in range(10):
    
    a = 0
    s_, r, done = env.step(s, a)
    s = s_

    print(r)