from networkx.algorithms import dominance
from graph_embedding import graph_embedding
from env import Env
from agent import PPO
import numpy as np
import networkx as nx
import torch
import scipy.sparse as sp

class PPOConfig:
    def __init__(self):
        self.env = Env()
        self.algo = 'PPO'
        # self.result_path = curr_path+"/results/" +self.env+'/'+curr_time+'/results/'  # path to save results
        # self.model_path = curr_path+"/results/" +self.env+'/'+curr_time+'/models/'  # path to save models
        self.train_eps = 200 # max training episodes
        # self.eval_eps = 50
        self.batch_size = 64
        self.gamma=0.99
        self.n_epochs = 4
        self.actor_lr = 0.0003
        self.critic_lr = 0.0003
        self.gae_lambda=0.95
        self.policy_clip=0.2
        self.hidden_dim = 2
        self.update_fre = 20 # frequency of agent update
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # check gpu

# def get_graph(G, s):
#     fea = G.nodes.data()
#     adj = np.array(nx.adjacency_matrix(G).todense())
#     print(fea, adj, s)
# r = 0
# get_graph(G, s)
#
# for i in range(10):
#
#     a = 0
#     s_, r, done = env.step(s, a)
#     s = s_
#
#     print(r)


def play(cfg, env, agent, adj):
    print('开始训练！')
    print(f'Env:{cfg.env}, Algorithm:{cfg.algo}, Device:{cfg.device}')
    rewards = []
    ma_rewards = []  # moving average rewards
    running_steps = 0
    for i_ep in range(cfg.train_eps):
        state = env.reset()
        # print(state)
        done = False
        ep_reward = 0
        while not done:
            action, prob, val = agent.choose_action(state, adj)
            # print(state)
            state_, reward, done = env.step(state, action)
            running_steps += 1
            ep_reward += reward
            agent.memory.push(state, action, prob, val, reward, done)
            if running_steps % cfg.update_fre == 0:
                agent.update(adj)
            state = state_
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(
                0.9 * ma_rewards[-1] + 0.1 * ep_reward)
        else:
            ma_rewards.append(ep_reward)
        print(f"回合：{i_ep + 1}/{cfg.train_eps}，奖励：{ep_reward:.2f}")
    print('Complete training！')
    return rewards, ma_rewards

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1), dtype=float)
    index = np.where(rowsum==0)
    rowsum[index] = 1.0
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

if __name__ == '__main__':
    # train
    # env,agent = env_agent_config(cfg,seed=1)
    # rewards, ma_rewards = train(cfg, env, agent)
    cfg = PPOConfig()
    env = cfg.env
    # state预处理得到observation的dim传到PPO中
    features = env.reset()
    features = sp.csr_matrix(features)
    features = normalize(features)#Integers to negative integer powers are not allowed.
    features = torch.FloatTensor(np.array(features.todense()))
    # features.reshape()
    # print(features.shape)
    G = graph_embedding(env, features)
    # print(G.edges.data())
    adj = np.array(nx.adjacency_matrix(G).todense())
    adj = sp.coo_matrix(adj)
    # print(adj)
    # adj = adj.tomatrix()
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)#'matrix' object has no attribute 'tocoo'

    # print(action_dim)
    agent = PPO(90, cfg)
    rewards, ma_rewards = play(cfg, env, agent, adj)

    # print(rewards)