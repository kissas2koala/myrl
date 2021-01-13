#
# coding=utf-8


import math
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from config import logger, GPU_CONFIG, IS_TEST
from ddpg.memo import ReplayMemory, Experience
from ddpg.model import Critic, Actor


class DDPG:
    def __init__(self, dim_obs, dim_act, actor_lr=0.001, critic_lr=0.01, gamma=0.9,
                 capacity=1000, batch_size=64, tau=0.01, hidden_size=64, device='cpu'):
        self.gamma = gamma
        self.memory = ReplayMemory(capacity)
        self.batch_size = batch_size
        self.tau = tau
        self.device = device
        self.use_cuda = GPU_CONFIG.use_cuda

        self.critic = Critic(dim_obs, dim_act, hidden_size)
        self.actor = Actor(dim_obs, dim_act, hidden_size)
        self.target_critic = Critic(dim_obs, dim_act, hidden_size)
        self.target_actor = Actor(dim_obs, dim_act, hidden_size)
        if self.use_cuda:
            self.critic = self.critic.cuda()
            self.actor = self.actor.cuda()
            self.target_critic = self.target_critic.cuda()
            self.target_actor = self.target_actor.cuda()




    def update(self):
        pass

    def select_action(self):
        pass