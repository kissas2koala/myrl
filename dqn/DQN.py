#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""
    DQN and DoubleDQN
"""

import math
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from config import logger, GPU_CONFIG, IS_TEST
from dqn.memo import ReplayMemory, Experience
from dqn.model import DQNModel


class DQN:
    """
    dqn or double dqn
    """
    def __init__(self, dim_obs, dim_act, lr=0.01, gamma=0.9, eps_high=0.95, eps_low=0.01, eps_decay=500,
                 batch_size=32, capacity=1000, is_ddqn=False, hidden_size=64):
        self.n_obs = dim_obs
        self.n_actions = dim_act

        self.lr = lr
        self.gamma = gamma
        self.capacity = capacity
        self.batch_size = batch_size
        self.epsilon = 0.
        self.eps_high = eps_high
        self.eps_low = eps_low
        self.eps_decay = eps_decay
        self.memory = ReplayMemory(capacity)
        self.target_replace = 100  # 网络替换的第二种方式
        self.scale_reward = 1
        self.is_ddqn = is_ddqn

        self.use_cuda = GPU_CONFIG.use_cuda
        self.device = 'gpu' if self.use_cuda else 'cpu'
        self.total_cnt = 0
        self.learn_cnt = 0
        self.loss_list = []  # [float, ..]
        self.grad_list = []  # [float, ..]

        # create policy and target net
        self.policy_net = DQNModel(self.n_obs, self.n_actions, hidden_size)
        self.target_net = DQNModel(self.n_obs, self.n_actions, hidden_size)
        # self.target_net = DQNModel(n_states, n_actions).to(self.device)  # 方法二
        if self.use_cuda:
            logger.info('GPU Available')
            self.policy_net = self.policy_net.cuda()
            self.target_net = self.target_net.cuda()
            # self.target_net = self.target_net.to(self.device)  # 方法二
        self.target_net.load_state_dict(self.policy_net.state_dict())
        # self.target_net.eval()  # 不启用 BatchNormalization 和 Dropout

        # loss
        self.loss_func = nn.MSELoss()
        # self.loss_func = F.smooth_l1_loss  # 计算Huber损失

        # optimizer
        self.optimizer = th.optim.Adam(self.policy_net.parameters(), lr=self.lr)

        self.FloatTensor = th.cuda.FloatTensor if self.use_cuda else th.FloatTensor
        self.LongTensor = th.cuda.LongTensor if self.use_cuda else th.LongTensor

    def learn(self):
        # check to replace target parameters  网络替换的第二种方式
        # if self.learn_cnt % self.target_replace == 0:
        #     self.target_net.load_state_dict(self.policy_net.state_dict())
        #     logger.debug('target_params_replaced')
        #     if IS_TEST:
        #         learn_cnt_str = '%09d' % self.learn_cnt
        #         th.save(self.target_net.state_dict(), 'model/dqn/model_' + learn_cnt_str + '.pkl')
        # self.learn_cnt += 1

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

        # q_eval w.r.t the action in experience
        q_eval = self.policy_net(obs_batch).gather(1, action_batch)  # shape (batch, 1)
        logger.debug(q_eval.detach())
        if self.is_ddqn:
            # use double dqn
            max_policy_action = self.policy_net(next_obs_batch).max(1)[1].detach().view(self.batch_size, 1)
            q_next = self.target_net(next_obs_batch).detach().gather(dim=1, index=max_policy_action)
        else:
            q_next = self.target_net(next_obs_batch).max(1)[0].detach()  # detach from graph, don't backpropagate
            q_next = q_next.view(self.batch_size, 1)
        q_target = self.scale_reward*reward_batch + self.gamma * q_next * (1-done_batch)  # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        # clip grad
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        # nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1) # 方法二
        # store grad
        param = next(self.policy_net.parameters())
        grad_mean = th.mean(param.grad.data)
        self.optimizer.step()

        # loss
        loss = loss.detach().cpu().numpy() if self.use_cuda else loss.detach().numpy()
        logger.debug('loss: {}'.format(loss))
        self.loss_list.append(loss)
        # grad_mean
        grad_mean = grad_mean.detach().cpu().numpy() if self.use_cuda else grad_mean.detach().numpy()
        self.grad_list.append(grad_mean)


    @th.no_grad()  # 不去计算梯度
    def select_action(self, obs):
        eps_high = self.eps_high
        eps_low = self.eps_low
        eps_decay = self.eps_decay
        self.epsilon = eps_low + (eps_high - eps_low) * (math.exp(-1.0 * self.total_cnt / eps_decay))
        self.total_cnt += 1
        # if np.random.uniform() < self.epsilon:
        if np.random.uniform() > self.epsilon:
            obs = th.unsqueeze(self.FloatTensor(obs), 0)  # (batch, )
            actions_value = self.policy_net(obs)
            action = th.max(actions_value, 1)[1]
            action = action.data.cpu() if self.use_cuda else action.detach()
            action = action.numpy()
        else:
            action = np.zeros(1, dtype=np.int32)
            action[0] = np.random.randint(0, self.n_actions)
        return action