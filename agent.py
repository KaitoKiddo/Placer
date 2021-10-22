import os
import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
# from model import Actor, Critic, BasicBlock
from model import Actor, Critic
from memory import PPOMemory
import random
import datetime

import os
import scipy.sparse as sp
import numpy as np
import torch
import torch.optim as optim
from model import Actor,Critic
from memory import PPOMemory


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1), dtype=float)
    index = np.where(rowsum == 0)
    rowsum[index] = 1.0
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

class PPO:
    def __init__(self, nfeat, cfg):
        self.gamma = cfg.gamma
        self.policy_clip = cfg.policy_clip
        self.n_epochs = cfg.n_epochs
        self.gae_lambda = cfg.gae_lambda
        self.device = cfg.device
        self.actor = Actor(nfeat, cfg.hidden_dim).to(self.device)
        self.critic = Critic(nfeat, cfg.hidden_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.memory = PPOMemory(cfg.batch_size)
        self.loss = 0

    def choose_action(self, observation, adj):
        # print(observation)
        # state = sp.csr_matrix(observation, dtype=np.float32)
        # print(state)
        # state = state.todense()
        # print(state)
        state = torch.tensor(observation, dtype=torch.float).to(self.device)
        adj = torch.tensor(adj, dtype=torch.float).to(self.device)
        # print(state)
        # state = torch.tensor(observation, dtype=torch.float)
        # state = Variable(torch.unsqueeze(state, dim=0).float(), requires_grad=False).to(self.device)
        dist = self.actor(state, adj)
        value = self.critic(state, adj)
        action = dist.sample()
        probs = torch.squeeze(dist.log_prob(action)).item()
        action = torch.squeeze(action).item()
        #在选中的框内选一个坐标
        col = action % 128
        row = int(action/128) + 1
        x = random.uniform((128-col)*75000, (129-col)*75000)
        y = random.uniform((128-row)*75000, (129-row)*75000)
        action = (x, y)
        print(action)
        value = torch.squeeze(value).item()
        return action, probs, value

    # def update(self):
    #     for _ in range(self.n_epochs):
    #         state_arr, action_arr, old_prob_arr, vals_arr,reward_arr, dones_arr, batches = self.memory.sample()
    #         values = vals_arr[:]
    #         ### compute advantage ###
    #         advantage = np.zeros(len(reward_arr), dtype=np.float32)
    #         for t in range(len(reward_arr)-1):
    #             discount = 1
    #             a_t = 0
    #             for k in range(t, len(reward_arr)-1):
    #                 a_t += discount*(reward_arr[k] + self.gamma*values[k+1]*\
    #                         (1-int(dones_arr[k])) - values[k])
    #                 discount *= self.gamma*self.gae_lambda
    #             advantage[t] = a_t
    #         advantage = torch.tensor(advantage).to(self.device)
    #         ### SGD ###
    #         values = torch.tensor(values).to(self.device)
    #         for batch in batches:
    #             states = torch.tensor(state_arr[batch], dtype=torch.float).to(self.device)
    #             old_probs = torch.tensor(old_prob_arr[batch]).to(self.device)
    #             actions = torch.tensor(action_arr[batch]).to(self.device)
    #             dist = self.actor(states)
    #             critic_value = self.critic(states)
    #             critic_value = torch.squeeze(critic_value)
    #             new_probs = dist.log_prob(actions)
    #             prob_ratio = new_probs.exp() / old_probs.exp()
    #             weighted_probs = advantage[batch] * prob_ratio
    #             weighted_clipped_probs = torch.clamp(prob_ratio, 1-self.policy_clip,
    #                     1+self.policy_clip)*advantage[batch]
    #             actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()
    #             returns = advantage[batch] + values[batch]
    #             critic_loss = (returns-critic_value)**2
    #             critic_loss = critic_loss.mean()
    #             total_loss = actor_loss + 0.5*critic_loss
    #             self.loss  = total_loss
    #             self.actor_optimizer.zero_grad()
    #             self.critic_optimizer.zero_grad()
    #             total_loss.backward()
    #             self.actor_optimizer.step()
    #             self.critic_optimizer.step()
    #     self.memory.clear()


    def update(self, adj):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr, \
            reward_arr, dones_arr, batches = \
                self.memory.sample()
            # print(reward_arr)
            values = vals_arr
            ### compute advantage ###
            advantage = np.zeros(len(reward_arr), dtype=np.float)
            for t in range(len(reward_arr) - 1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr) - 1):
                    a_t += discount * (reward_arr[k] + self.gamma * values[k + 1] * \
                                       (1 - int(dones_arr[k])) - values[k])
                    discount *= self.gamma * self.gae_lambda
                advantage[t] = a_t
            advantage = torch.tensor(advantage).to(self.device)
            ### SGD ###
            values = torch.tensor(values).to(self.device)
            for batch in batches:
                print("batch is:"+str(batch))
                states = torch.tensor(state_arr[batch], dtype=torch.float).to(self.device)
                # states = state_arr[batch]
                states = torch.squeeze(states, 1)
                # print(states)
                # ###
                # # states = sp.csr_matrix(states)
                # # states = normalize(states)  # Integers to negative integer powers are not allowed.
                # states = torch.FloatTensor(np.array(states.todense()))
                old_probs = torch.tensor(old_prob_arr[batch]).to(self.device)
                actions = torch.tensor(action_arr[batch]).to(self.device)
                dist = self.actor(states, adj)
                critic_value = self.critic(states, adj)
                critic_value = torch.squeeze(critic_value)
                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1 - self.policy_clip,
                                                     1 + self.policy_clip) * advantage[batch]
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()
                returns = advantage[batch] + values[batch]
                critic_loss = (returns - critic_value) ** 2
                critic_loss = critic_loss.mean()
                total_loss = actor_loss + 0.5 * critic_loss
                self.loss = total_loss
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()
        self.memory.clear()
        # def save(self,path):
    #     actor_checkpoint = os.path.join(path, str(self.env)+'_actor.pt')
    #     critic_checkpoint= os.path.join(path, str(self.env)+'_critic.pt')
    #     torch.save(self.actor.state_dict(), actor_checkpoint)
    #     torch.save(self.critic.state_dict(), critic_checkpoint)
    # def load(self,path):
    #     actor_checkpoint = os.path.join(path, str(self.env)+'_actor.pt')
    #     critic_checkpoint= os.path.join(path, str(self.env)+'_critic.pt')
    #     self.actor.load_state_dict(torch.load(actor_checkpoint))
    #     self.critic.load_state_dict(torch.load(critic_checkpoint))


