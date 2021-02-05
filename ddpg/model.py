#
# coding=utf-8


import torch as th
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    def __init__(self, dim_obs, dim_acttion, hidden_size):
        super(Critic, self).__init__()
        self.linear_1 = nn.Linear(dim_obs+dim_acttion, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.linear_3 = nn.Linear(hidden_size, 1)

        # 随机初始化较小的值
        init_w = 3e-3
        self.linear_3.weight.data.uniform_(-init_w, init_w)
        self.linear_3.bias.data.uniform_(-init_w, init_w)

    def forward(self, obs, action):
        x = th.cat((obs, action), dim=1)
        x = F.relu(self.linear_1(x))
        # x = F.relu(self.linear_2(x))
        out = self.linear_3(x)
        return out


class Actor(nn.Module):
    def __init__(self, dim_obs, dim_action, hidden_size):
        super(Actor, self).__init__()
        self.linear_1 = nn.Linear(dim_obs, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.linear_3 = nn.Linear(hidden_size, dim_action)

        # 随机初始化较小的值
        init_w = 3e-3
        self.linear_3.weight.data.uniform_(-init_w, init_w)
        self.linear_3.bias.data.uniform_(-init_w, init_w)

    def forward(self, obs):
        x = F.relu(self.linear_1(obs))
        # x = F.relu(self.linear_2(x))
        out = F.tanh(self.linear_3(x))
        return out
