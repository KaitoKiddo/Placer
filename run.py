from networkx.readwrite import graph6
from env import Env

env = Env()

s = env.reset()
print(s)
r = 0
for i in range(5):
    
    a = 0
    s_, reward = env.step(s, a)
    s = s_
    r += reward

print(r)