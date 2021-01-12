#
# coding=utf-8

import torch as th
import torch.nn as nn


class DQNModel(nn.Module):
    def __init__(self, dim_obs, dim_action, hidden_units_num):
        super(DQNModel, self).__init__()
        self.linear_1 = nn.Linear(dim_obs, hidden_units_num)
        self.linear_2 = nn.Linear(hidden_units_num, hidden_units_num)
        self.linear_out = nn.Linear(hidden_units_num, dim_action)

        self.LReLU = nn.LeakyReLU(0.01)
        self.ReLU = nn.ReLU()
        self.Tanh = nn.Tanh()

        # self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('leaky_relu')
        nn.init.xavier_uniform_(self.linear_1.weight, gain=gain)
        nn.init.xavier_uniform_(self.linear_2.weight, gain=gain)
        nn.init.xavier_uniform_(self.linear_out.weight, gain=gain)

    def forward(self, x):
        x = self.ReLU(self.linear_1(x))
        x = self.ReLU(self.linear_2(x))
        out = self.linear_out(x)
        return out
