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


def file_w(lst, file_name, file_path='results/'):
    """
    :param reward: float
    :param file_name: string
    :param file_path: string
    :return:
    """
    s = ','.join([str(ele) for ele in lst])
    with open(file_path+file_name, 'w') as f:
        f.write(s)


def file_r(file_name, file_path='results/'):
    """
    :param file_name: string
    :param file_path: string
    :return: list(float)
    """
    with open(file_path+file_name, 'r') as f:
        s = f.read()
    content_list = [float(ele) for ele in s.split(',')]
    return content_list


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
    parser.add_argument('--pics', type=str, default='', help='input name need to make pics')
    args = parser.parse_args()
    # reward
    if args.pics == 'reward':
        lst = file_r('reward.txt')
        make_pic(lst, 'reward')
    elif args.pics == 'moving_average_reward':
        lst = file_r('moving_average_reward.txt')
        make_pic(lst, 'moving_average_reward')
    # loss
    elif args.pics == 'loss':
        lst = file_r('loss.txt')
        make_pic(lst, 'loss')
    elif args.pics == 'a_loss':
        lst = file_r('a_loss.txt'.format(PICS_PATH))
        make_pic(lst, 'a loss')
    elif args.pics == 'c_loss':
        lst = file_r('c_loss.txt'.format(PICS_PATH))
        make_pic(lst, 'c loss')
