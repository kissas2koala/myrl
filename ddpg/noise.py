#
# coding=utf-8


import numpy as np


class OUNoise(object):
    """
        OU noise
    """

    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.n_actions = action_space.shape[0]
        self.low = action_space.low
        self.high = action_space.high
        self.reset()

    def reset(self):
        self.obs = np.ones(self.n_actions) * self.mu

    def evolve_obs(self):
        x = self.obs
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.n_actions)
        self.obs = x + dx
        return self.obs

    def get_action(self, action, t=0):
        ou_obs = self.evolve_obs()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_obs, self.low, self.high)


class GaussianNoise:
    """
        Gaussian noise
    """

    def __init__(self, action_space, min_var=0.005, decay=500):
        self.n_action = action_space.shape[0]
        self.low = action_space.low
        self.high = action_space.high
        self.var = 1
        self.var_origin = self.var
        self.min_var = min_var
        self.decay = decay

    def reset(self):
        pass

    def update_var(self, step_i):
        if self.var > self.min_var:
            # shrink = step_i * (self.min_var//self.decay)
            shrink = 0.998
            self.var *= shrink

    def get_action(self, action, step_i):
        noise = np.random.randn(self.n_action) * self.var
        # self.update_var(step_i)
        return np.clip(action + noise, self.low, self.high)
