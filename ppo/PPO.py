#
# coding=utf-8


import numpy as np
import torch as th
import torch.nn as nn

from config import GPU_CONFIG, logger
from ppo.model import Actor, Critic


class Buffer:
    def __init__(self):
        self.buffer_s = []
        self.buffer_a = []
        self.buffer_r = []
        self.buffer_s_ = []
        self.buffer_done = []
        self.size = 0

    def add(self, s, a, r, s_, done):
        self.buffer_s.append(s)
        self.buffer_a.append(a)
        self.buffer_r.append(r)
        self.buffer_s_.append(s_)
        self.buffer_done.append(done)
        self.size += 1

    def clear(self):
        self.buffer_s.clear()
        self.buffer_a.clear()
        self.buffer_r.clear()
        self.buffer_s_.clear()
        self.buffer_done.clear()
        self.size = 0


class PPO:
    def __init__(self, dim_obs, dim_act, actor_lr=0.001, critic_lr=0.01,gamma=0.9,
                 clip_epsilon=0.1, a_update_epochs=5, c_update_epochs=5, hidden_size=32,
                 buffer_size=32):
        self.n_obs = dim_obs
        self.n_actions = dim_act

        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.a_update_epochs = a_update_epochs
        self.c_update_epochs = c_update_epochs
        self.hidden_size = hidden_size

        self.buffer_size = buffer_size
        self.buffer = Buffer()
        self.device = 'cpu'

        self.critic = Critic(dim_obs, hidden_size).to(self.device)
        self.actor = Actor(dim_obs, dim_act, hidden_size).to(self.device)
        self.old_actor = Actor(dim_obs, dim_act, hidden_size).to(self.device)
        self.old_actor.load_state_dict(self.actor.state_dict())

        self.critic_optimizer = th.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.actor_optimizer = th.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.FloatTensor = th.cuda.FloatTensor if GPU_CONFIG.use_cuda else th.FloatTensor
        self.LongTensor = th.cuda.LongTensor if GPU_CONFIG.use_cuda else th.LongTensor

    def update(self):
        # pre data
        buffer_s = self.FloatTensor(np.array(self.buffer.buffer_s))
        buffer_a = self.LongTensor(np.array(self.buffer.buffer_a)).view(self.buffer_size, 1)
        buffer_s_ = self.FloatTensor(np.array(self.buffer.buffer_s_))
        # pre reward
        buffer_r = []
        discounted_reward = 0.
        for i in range(len(self.buffer.buffer_r)-1, -1, -1):
            if self.buffer.buffer_done[i]:
                discounted_reward = 0.
            discounted_reward = self.buffer.buffer_r[i] + (self.gamma * discounted_reward)
            buffer_r.append(discounted_reward)
        buffer_r.reverse()
        buffer_r = self.FloatTensor(np.array(buffer_r)).view(self.buffer_size, 1)
        # a_loss
        value = self.critic(buffer_s).detach()
        advantage = buffer_r - value
        old_acts_prob = self.old_actor(buffer_s).detach()
        for _ in range(self.a_update_epochs):
            acts_prob = self.actor(buffer_s)
            ratio = acts_prob.gather(1, buffer_a) / old_acts_prob.gather(1, buffer_a)
            surr1 = ratio*advantage
            surr2 = th.clamp(ratio, 1-self.clip_epsilon, 1+self.clip_epsilon)*advantage
            a_loss = -th.min(surr1, surr2).mean()
            self.actor_optimizer.zero_grad()
            a_loss.backward()
            self.actor_optimizer.step()
        # c_loss
        for _ in range(self.c_update_epochs):
            value = self.critic(buffer_s)
            c_loss = nn.MSELoss()(value, buffer_r)
            self.critic_optimizer.zero_grad()
            c_loss.backward()
            self.critic_optimizer.step()
        # policy to old_policy
        self.old_actor.load_state_dict(self.actor.state_dict())
        return a_loss, c_loss

    @th.no_grad()
    def choose_action(self, obs):
        obs = th.unsqueeze(self.FloatTensor(obs), 0)
        acts_prob = self.actor(obs)
        acts_prob = acts_prob.data.cpu() if GPU_CONFIG.use_cuda else acts_prob.detach()
        acts_prob = acts_prob.numpy()
        action = np.random.choice(np.arange(acts_prob.shape[1]), p=acts_prob.ravel())
        return action
