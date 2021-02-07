#
# coding=utf-8

from qlearning.QLearning import QLearning


if __name__ == "__main__":
    agent = QLearning(dim_obs, dim_act)

    for i_episode in max_episodes:
        s = env.reset()

        for i_step in max_steps:
            a = agent.get_action(s)
            s_, r, done = env.step(a)

            agent.learn(s, a, r, s_, done)

            s = s_
            if done:
                break