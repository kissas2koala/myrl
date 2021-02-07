#
# coding=utf-8

import numpy as np


class Sarsa:

    def __init__(self, obs_dim, act_dim, lr=0.01, gamma=0.9):
        self.q_table = np.zeros((obs_dim, act_dim), dtype=np.float)
        self.lr = lr
        self.gamma = gamma

    def learn(self, s, a, r, s_, a_, done):
        q = self.q_table[s, a]
        q_target = r if done else r + self.gamma * self.q_table[s_, a_]
        self.q_table[s, a] += self.lr * (q_target - q)

    def get_action(self, s):
        pass