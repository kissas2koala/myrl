#
# coding=utf-8

import torch as th
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    def __init__(self, dim_obs, hidden_size):
        super(Critic, self).__init__()
        self.linear_1 = nn.Linear(dim_obs, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.linear_3 = nn.Linear(hidden_size, 1)

        # 随机初始化较小的值
        # init_w = 3e-3
        # self.linear_3.weight.data.uniform_(-init_w, init_w)
        # self.linear_3.bias.data.uniform_(-init_w, init_w)

    def forward(self, obs):
        x = obs
        x = F.tanh(self.linear_1(x))
        x = F.tanh(self.linear_2(x))
        out = self.linear_3(x)
        return out


class Actor(nn.Module):
    def __init__(self, dim_obs, dim_action, hidden_size, is_continuous=False):
        super(Actor, self).__init__()
        self.is_continuous = is_continuous

        self.linear_1 = nn.Linear(dim_obs, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.linear_3 = nn.Linear(hidden_size, dim_action)

        # 随机初始化较小的值
        # init_w = 3e-3
        # self.linear_3.weight.data.uniform_(-init_w, init_w)
        # self.linear_3.bias.data.uniform_(-init_w, init_w)

    def forward(self, obs):
        x = F.tanh(self.linear_1(obs))
        x = F.tanh(self.linear_2(x))
        # 判断上是否为连续动作
        if self.is_continuous:
            mu = F.tanh(self.linear_3(x))
            sigma = F.softplus(self.linear_3(x))
            out = (mu, sigma)
        else:
            out = F.softmax(self.linear_3(x), dim=1)
        return out
