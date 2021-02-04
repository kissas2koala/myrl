#
# coding=utf-8


import time
import argparse
import numpy as np
import gym

from ddpg.DDPG import DDPG
from ddpg.env import NormalizedActions
from ddpg.noise import OUNoise, GaussianNoise
from config import logger, IS_TEST
from utils import file_w


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=1, type=int)  # 1 表示训练，0表示只进行eval
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--tau", default=0.01, type=float)

    parser.add_argument("--critic_lr", default=1e-3, type=float)
    parser.add_argument("--actor_lr", default=1e-4, type=float)
    parser.add_argument("--hidden_size", default=128, type=int)
    parser.add_argument("--memory_capacity", default=10000, type=int, help="capacity of Replay Memory")
    parser.add_argument("--batch_size", default=64, type=int, help="batch size of memory sampling")

    parser.add_argument("--max_eps", default=200, type=int)  # 训练的最大episode数目
    parser.add_argument("--max_steps", default=200, type=int)
    params = parser.parse_args()

    return params


def main(params):
    t1 = time.time()
    # environment
    env = gym.make("Pendulum-v0")
    # env = NormalizedActions(env)
    env.seed(1)
    ou_noise = OUNoise(env.action_space)
    n_states = env.observation_space.shape[0]  # (4, )
    n_actions = env.action_space.shape[0]  # 1 -2<x<2, changed: -1<x<1
    logger.info("obs num: %d" % n_states)
    logger.info("act num: %d" % n_actions)
    logger.info("act high: %d" % env.action_space.high)
    logger.info("act low: %d" % env.action_space.low)

    RL = DDPG(dim_obs=n_states, dim_act=n_actions, actor_lr=params.actor_lr, critic_lr=params.critic_lr,
              gamma=params.gamma,capacity=params.memory_capacity, batch_size=params.batch_size, tau=params.tau,
              hidden_size=params.hidden_size)

    # execution
    total_rewards = []
    moving_average_rewards = []
    moving_average_reward = 0.0
    a_loss_list = []
    c_loss_list = []
    total_cnt = 0
    for i_episode in range(1, params.max_eps + 1):
        total_reward = 0.0  # 每回合所有智能体的总体奖励
        # eps_r = 0.
        obs = env.reset()
        ou_noise.reset()
        for i_step in range(params.max_steps):
            # env.render()
            act = RL.select_action(obs)
            # logger.info("select act: {}".format(act))
            action = np.squeeze(act)
            action =ou_noise.get_action(action, i_step)
            # logger.info("select action: {}".format(act))
            # action = env.action_space.sample()
            next_obs, r, done, _ = env.step(action)
            total_reward += r
            # if done: r=0
            RL.memory.push(obs, action, next_obs, r*100, done)
            obs = next_obs

            if total_cnt > params.batch_size:
                a_loss, c_loss = RL.learn()
                a_loss_list.append(a_loss)
                c_loss_list.append(c_loss)

            if done:
                break
            total_cnt += 1

        moving_average_reward = total_reward if i_episode == 1 else moving_average_reward * 0.9 + total_reward * 0.1
        moving_average_rewards.append(moving_average_reward)
        total_rewards.append(total_reward)
        logger.info('episode:{}, reward:{}, step:{}'.format(i_episode, total_reward, i_step + 1))
    if not IS_TEST:
        label = '' or 'r10'
        file_w(total_rewards, 'reward@{}@.txt'.format(label))
        file_w(moving_average_rewards, 'moving_average_reward@{}@.txt'.format(label))
        file_w(a_loss_list, 'a_loss@{}@.txt'.format(label))
        file_w(c_loss_list, 'c_loss@{}@.txt'.format(label))

    env.close()
    t2 = time.time()
    logger.info('**********train finish!**************, time:%f' % (t2 - t1))


if __name__ == '__main__':
    params = get_params()
    main(params)
