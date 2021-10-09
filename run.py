from graph_embedding import graph_embedding
from networkx.readwrite import graph6
from env import Env

env = Env()

s = env.reset()
G = graph_embedding(env)
r = 0
for i in range(10):
    
    a = 0
    s_, reward = env.step(s, a)
    s = s_
    r += reward

print(r)