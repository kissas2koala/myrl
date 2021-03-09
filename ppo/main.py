#
# coding=utf-8


import time
import argparse
import gym
from ppo.PPO import PPO
from config import logger, IS_TEST
from utils import file_w


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=1, type=int)  # 1 表示训练，0表示只进行eval
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--clip_epsilon", defalut=0.2, type=float)
    parser.add_argument("--a_update_epochs", defalut=5, type=int)
    parser.add_argument("--c_update_epochs", defalut=5, type=int)
    parser.add_argument("--buffer_size", default=32, type=int)

    parser.add_argument("--critic_lr", default=2e-4, type=float)
    parser.add_argument("--actor_lr", default=1e-4, type=float)
    parser.add_argument("--hidden_size", default=32, type=int)

    parser.add_argument("--max_eps", default=300, type=int)  # 训练的最大episode数目
    parser.add_argument("--max_steps", default=200, type=int)
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

    RL = PPO(dim_obs=n_states, dim_act=n_actions, actor_lr=params.actor_lr, critic_lr=params.critic_lr,
             gamma=params.gamma, clip_epsilon=params.clip_epsilon, a_update_epochs=params.a_update_epochs,
             c_update_epochs=params.c_update_epochs, hidden_size=params.hidden_size, buffer_size=params.buffer_size)

    # execution
    total_rewards = []
    moving_average_rewards = []
    moving_average_reward = 0.0
    total_cnt = 0
    for i_episode in range(params.max_eps):
        total_reward = 0.0
        s = env.reset()
        for i_step in range(params.max_steps):
            action = RL.choose_action(s)
            s_, r, done, _ = env.step(action)
            total_reward += r

            RL.buffer.add(s, action, r, s_, done)

            if RL.buffer.size >= params.batch_size:
                RL.update()
                RL.buffer.clear()

            s = s_
            if done:
                break
            total_cnt += 1

        moving_average_reward = total_reward if i_episode == 1 else moving_average_reward * 0.9 + total_reward * 0.1
        moving_average_rewards.append(moving_average_reward)
        total_rewards.append(total_reward)
        logger.info('episode:{}, reward:{} step:{}'.format(i_episode, total_reward, i_step + 1))
    if not IS_TEST:
        label = '' or ''
        # file_w(total_rewards, 'reward@{}@.txt'.format(label))
        file_w(moving_average_rewards, 'moving_average_reward@{}@.txt'.format(label))
        # file_w(a_loss_list, 'a_loss@{}@.txt'.format(label))
        # file_w(c_loss_list, 'c_loss@{}@.txt'.format(label))

    env.close()
    t2 = time.time()
    logger.info('**********train finish!**************, time:%f' % (t2 - t1))


if __name__ == '__main__':
    params = get_params()
    main(params)
