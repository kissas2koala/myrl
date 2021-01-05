#
# coding=utf-8


import time
import argparse
import numpy as np
import gym

from dqn.DQN import DQN
from config import logger, IS_TEST
from utils import reward_w, loss_w


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=1, type=int)  # 1 表示训练，0表示只进行eval
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--eps_high", default=0.95, type=float)  # 基于贪心选择action对应的参数epsilon
    parser.add_argument("--eps_low", default=0.01, type=float)
    parser.add_argument("--eps_decay", default=500, type=float)
    parser.add_argument("--policy_lr", default=0.001, type=float)
    parser.add_argument("--memory_capacity", default=1000, type=int, help="capacity of Replay Memory")

    parser.add_argument("--batch_size", default=32, type=int,
                        help="batch size of memory sampling")
    parser.add_argument("--train_eps", default=200, type=int)  # 训练的最大episode数目
    parser.add_argument("--train_steps", default=200, type=int)
    parser.add_argument("--target_replace", default=2, type=int,
                        help="when(every default 2 eisodes) to update target net ")  # 更新频率

    parser.add_argument("--eval_eps", default=100, type=int)
    parser.add_argument("--eval_steps", default=200, type=int)
    params = parser.parse_args()

    return params


def main(params):
    t1 = time.time()
    # environment
    env = gym.make('CartPole-v0')
    env.seed(1)
    n_states = env.observation_space.shape[0]  # (4, )
    n_actions = env.action_space.n  # 2
    logger.info("obs num: %d" % n_states)
    logger.info("act num: %d" % n_actions)

    RL = DQN(dim_obs=n_states, dim_act=n_actions, lr=params.policy_lr, gamma=params.gamma,
             eps_high=params.eps_high, eps_low=params.eps_low, eps_decay=params.eps_decay,
             capacity=params.memory_capacity, batch_size=params.batch_size, target_replace=params.target_replace)

    # execution
    total_cnt = 0
    moving_average_reward = 0
    for i_episode in range(1, params.train_eps + 1):
        step_cnt = 0
        total_reward = 0.0  # 每回合所有智能体的总体奖励
        # eps_r = 0.
        obs = env.reset()
        while True:
            # env.render()
            act = RL.select_action(obs)
            # logger.info("selet action: ", act)
            action = np.squeeze(act)
            next_obs, r, done, _ = env.step(action)
            total_reward += r
            # if done: r=0
            RL.memory.push(obs, act, next_obs, r, done)
            obs = next_obs

            if total_cnt > params.batch_size:
                loss = RL.learn()
                if not IS_TEST: loss_w(loss, 'loss.txt')

            if done:
                break

        if i_episode % params.target_replace == 0:
            RL.target_net.load_state_dict(RL.policy_net.state_dict())

        step_cnt += 1
        total_cnt += 1
        moving_average_reward = total_reward if i_episode == 1 else moving_average_reward * 0.9 + total_reward * 0.1
        logger.info('episode:{}, reward:{}, e_greedy:{:.2f}, step:{}'.
                    format(i_episode, total_reward, RL.epsilon, step_cnt + 1))
        if not IS_TEST:
            reward_w(total_reward, 'reward.txt')
            reward_w(moving_average_reward, 'moving_average_reward.txt')

    env.close()
    t2 = time.time()
    logger.info('**********train finish!**************, time:%f' % (t2 - t1))


if __name__ == '__main__':
    params = get_params()
    main(params)
