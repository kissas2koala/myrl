#
# coding=utf-8

import numpy as np
import torch as th
import torch.nn as nn
from a2c.model import Critic, Actor
from config import GPU_CONFIG, logger


class A2C:
    def __init__(self, dim_obs, dim_act, actor_lr=0.01, critic_lr=0.001, gamma=0.9, hidden_size=64):
        self.n_obs = dim_obs
        self.n_actions = dim_act

        self.gamma = gamma
        self.hidden_size = hidden_size

        self.device = 'cpu'

        self.critic = Critic(dim_obs, hidden_size).to(self.device)
        self.actor = Actor(dim_obs, dim_act, hidden_size).to(self.device)

        self.critic_optimizer = th.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.actor_optimizer = th.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.FloatTensor = th.cuda.FloatTensor if GPU_CONFIG.use_cuda else th.FloatTensor
        self.LongTensor = th.cuda.LongTensor if GPU_CONFIG.use_cuda else th.LongTensor

    def update(self, s, a, r, s_, done):
        # pre data
        logger.debug(s, a, r, s_, done)
        s = th.unsqueeze(self.FloatTensor(np.array(s)), 0)
        a = th.unsqueeze(self.LongTensor(np.array([a])), 0)
        r = th.unsqueeze(self.FloatTensor(np.array([r])), 0)
        s_ = th.unsqueeze(self.FloatTensor(np.array(s_)), 0)
        done = th.unsqueeze(self.FloatTensor(np.array([done])), 0)
        logger.debug(s, a, r, s_, done)
        # c_loss
        value = self.critic(s)
        value_target = r if done else r + self.gamma * self.critic(s_)
        advantage = value_target - value  # TD_error = (r+gamma*V_next) - V_eval
        c_loss = th.pow(advantage, 2)
        self.critic_optimizer.zero_grad()
        c_loss.backward()
        self.critic_optimizer.step()
        # a_loss
        acts_prob = self.actor(s)
        a_loss = advantage.detach() * th.log(acts_prob.gather(1, a))
        self.actor_optimizer.zero_grad()
        a_loss.backward()
        self.actor_optimizer.step()
        return a_loss, c_loss

    @th.no_grad()
    def choose_action(self, obs):
        obs = th.unsqueeze(self.FloatTensor(obs), 0)
        acts_prob = self.actor(obs)
        acts_prob = acts_prob.data.cpu() if GPU_CONFIG.use_cuda else acts_prob.detach()
        acts_prob = acts_prob.numpy()
        action = np.random.choice(np.arange(acts_prob.shape[1]), p=acts_prob.ravel())
        return action
