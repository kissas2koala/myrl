#
# coding=utf-8

from sarsa.Sarsa import Sarsa


if __name__ == "__main__":
    agent = Sarsa(dim_obs, dim_act)

    for i_episode in max_episodes:
        s = env.reset()
        a = agent.get_action(s)

        for i_step in max_steps:
            s_, r, done = env.step(a)
            a_ = agent.get_action(s_)

            agent.learn(s, a, r, s_, a_, done)

            s = s_
            a = a_
            if done:
                break