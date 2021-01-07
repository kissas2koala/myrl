#
# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
import torch as th
import argparse

from config import PICS_PATH


def norm(x):
    """
    归一化
    :param x: tensor
    :return: ret: tensor
    """
    x_max = th.max(x)
    x_min = th.min(x)
    ret = (x - x_min) / (x_max - x_min)
    return ret


def norm_np(x):
    """
    归一化
    :param x: ndarray
    :return: ret: ndarray
    """
    x_max = np.max(x)
    x_min = np.min(x)
    ret = (x - x_min) / (x_max - x_min)
    return ret


def std(x):
    """
    :param x: tensor
    :return: ret: tensor
    """
    mu = th.mean(x)
    std = th.std(x)
    ret = (x - mu) / std
    return ret


def reward_w(reward, file_name, file_path='results/'):
    """
    将得分写入文件
    :param reward: float
    :param file_name: string
    :return:
    """
    score_string = '%.2f,' % reward
    with open(file_path+file_name, 'a') as f:
        f.write(score_string)


def reward_r(file_name, file_path='results/'):
    """
    :param file_name: string
    :return: list(float)
    """
    with open(file_path+file_name, 'r') as f:
        reward_string = f.read()
    reward_list = [float(ele) for ele in reward_string.split(',')[:-1]]
    return reward_list


def loss_w(loss, file_name, file_path='results/'):
    """
    :param loss: float
    :param file_name: string
    :return:
    """
    loss_string = '%.2f,' % loss
    with open(file_path+file_name, 'a') as f:
        f.write(loss_string)


def loss_r(file_name, file_path='results/'):
    """
    :param file_name: string
    :return: list(float)
    """
    with open(file_path+file_name, 'r') as f:
        loss_string = f.read()
    loss_list = [float(ele) for ele in loss_string.split(',')[:-1]]
    return loss_list


def make_pic(lst, title_name):
    """
    :param lst: list(float)
    :param title_name: string
    :return:
    """
    x = np.arange(1, len(lst) + 1)
    print("length: {}".format(len(lst)))
    y = np.array(lst)
    plt.title(title_name)
    plt.plot(x, y)
    plt.show()


if __name__ == '__main__':
    # 创建解析器
    parser = argparse.ArgumentParser()
    parser.add_argument('--pics', type=str, default='', help='input sth')
    args = parser.parse_args()

    # reward
    if args.pics == 'reward':
        lst = reward_r('reward.txt'.format(PICS_PATH))
        make_pic(lst, 'reward')
    # loss
    elif args.pics == 'loss':
        lst = loss_r('loss.txt')
        make_pic(lst, 'loss')
    elif args.pics == 'a_loss':
        lst = loss_r('a_loss.txt'.format(PICS_PATH))
        make_pic(lst, 'a loss')
    elif args.pics == 'c_loss':
        lst = loss_r('c_loss.txt'.format(PICS_PATH))
        make_pic(lst, 'c loss')
