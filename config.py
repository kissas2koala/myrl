#
# coding=utf-8


import os
import torch as th


class Logger(object):
    def __init__(self, level):
        if level == 'info':
            self.level = 1
        elif level == 'debug':
            self.level = 0
        elif level == 'danger':
            self.level = 2
        else:
            raise (Exception('level params error!'))

    def info(self, *args):
        if self.level < 2:
            print(', '.join([str(ele) for ele in args]))

    def debug(self, *args):
        if self.level < 1:
            # msg = msg if msg else 'wait!'
            print(', '.join([str(ele) for ele in args]))
            # input('pause: input sth and enter: ')

    def warn(self, *args):
        if self.level < 3:
            print(', '.join([str(ele) for ele in args]))

    def wait(self):
        input('wait! pause: input sth and enter: ')


class GPUConfig(object):
    def __init__(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # todo 一定要放在th.cuda前面
        self.use_cuda = False  # th.cuda.is_available()
        self.use_parallel = False
        self.device_ids = [0]
        if self.use_cuda:
            # 分配gpu
            # th.cuda.set_device(1)
            self.device = th.device("cuda:0")  # 指定模型训练所在 GPU
            if self.use_parallel:
                th.distributed.init_process_group(backend='nccl')
                # 需要用此命令运行 python -m torch.distributed.launch main.py
        else:
            self.device = 'cpu'


# 日志单例
logger = Logger(level="info")
# gpu配置
GPU_CONFIG = GPUConfig()

# model agent name
MODEL_NAME = ''
# model store path
MODEL_PATH = ''
# pics path or ''
PICS_PATH = ''

# todo is test
IS_TEST = False

if __name__ == '__main__':
    # logger.debug("debug")
    pass
