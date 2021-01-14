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


def soft_update(target, source, tau):
    for target_param, source_param in zip(target.parameters(),
                                          source.parameters()):
        target_param.data.copy_(
            (1. - tau) * target_param.data + tau * source_param.data)


class DDPG:
    def __init__(self, dim_obs, dim_act, actor_lr=0.001, critic_lr=0.01, gamma=0.9,
                 capacity=1000, batch_size=64, tau=0.01, hidden_size=64):
        self.gamma = gamma
        self.memory = ReplayMemory(capacity)
        self.batch_size = batch_size
        self.tau = tau
        self.device = 'gpu' if GPU_CONFIG.use_cuda else 'cpu'
        self.learn_cnt = 0
        self.FloatTensor = th.cuda.FloatTensor if GPU_CONFIG.use_cuda else th.FloatTensor

        self.critic = Critic(dim_obs, dim_act, hidden_size).to(self.device)
        self.actor = Actor(dim_obs, dim_act, hidden_size).to(self.device)
        self.target_critic = Critic(dim_obs, dim_act, hidden_size).to(self.device)
        self.target_actor = Actor(dim_obs, dim_act, hidden_size).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_actor.load_state_dict(self.actor.state_dict())
        # for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
        #     target_param.data.copy_(param.data)  # 方式二: todo 试试

        self.critic_optimizer = th.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.actor_optimizer = th.optim.Adam(self.actor.parameters(), lr=actor_lr)

    def update(self):
        # sample batch from all memory
        transitions = self.memory.sample(self.batch_size)
        batch = Experience(*zip(*transitions))  # class(list)
        obs_batch = self.FloatTensor(np.array(batch.obs))  # torch.Size([32, 4])
        # obs_batch = th.tensor(obs_batch, device=self.device, dtype=th.float)  # 方法二
        logger.debug("obs_batch: {}".format(obs_batch.shape))
        action_batch = self.LongTensor(np.array(batch.action))  # (batch, 1)
        logger.debug('action batch: {}'.format(action_batch.shape))
        reward_batch = self.FloatTensor(np.array(batch.reward)).view(self.batch_size, 1)  # (batch, 1)
        logger.debug('reward_batch: {}'.format(reward_batch.shape))
        next_obs_batch = self.FloatTensor(np.array(batch.next_obs))
        done_batch = self.FloatTensor(np.array(batch.done)).view(self.batch_size, 1)  # (batch, 1)
        logger.debug('done_batch: {}'.format(done_batch))

        # c loss
        self.critic_optimizer.zero_grad()
        q_eval = self.critic(obs_batch, action_batch)
        next_action = self.target_actor(next_obs_batch).detach()
        q_next = self.target_critic(next_obs_batch, next_action).detach()
        q_target = reward_batch + self.gamma * q_next * (1 - done_batch)
        c_loss = nn.MSELoss()(q_eval, q_target)
        c_loss.backward()
        self.critic_optimizer.step()

        # a loss
        self.actor_optimizer.zero_grad()
        current_action = self.actor(obs_batch)
        policy_loss = self.critic(obs_batch, current_action)
        a_loss = -policy_loss.mean()
        a_loss.backward()
        self.actor_optimizer.step()

        # soft update actor_target, critic_target
        soft_update(self.target_critic, self.critic, self.tau)
        soft_update(self.target_actor, self.actor, self.tau)

    @th.no_grad()
    def select_action(self, obs):
        obs = self.FloatTensor(obs).unsqueeze(0)
        action = self.actor(obs).detach()
        action = action.cpu().numpy() if GPU_CONFIG.use_cuda else action.numpy()
        return action
