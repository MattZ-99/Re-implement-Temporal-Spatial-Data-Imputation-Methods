import torch
from torch import optim
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import time


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def make_scheduler(optimizer, lr, min_lr, epochs, steps=10):
    if min_lr < 0:
        return None
    step_size = epochs // steps
    gamma = (min_lr / lr)**(1 / steps)
    return optim.lr_scheduler.StepLR(
        optimizer, step_size=step_size, gamma=gamma)


class ValueStat:
    def __init__(self):
        self.value = 0
        self.count = 0

    def update(self, v=0, n=1):
        self.value += v * n
        self.count += n

    def reset(self):
        self.value = 0
        self.count = 0

    def get_sum(self):
        return self.value

    def get_avg(self):
        if self.count == 0:
            return -1
        return self.value/self.count


def mask_data(data, mask, tau):
    return mask * data + (1 - mask) * tau


def makedirs(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def time_for_save():
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


class ValuesVisual:
    def __init__(self):
        self.values = []

    def add_value(self, val):
        self.values.append(val)

    def __len__(self):
        return len(self.values)

    def plot(self, output_path, title="Title", xlabel="X-axis", ylabel="Y-axis"):
        length = len(self)
        if length == 0:
            return -1
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        x_axis = [i for i in range(length)]
        ax.plot(x_axis, self.values, color='tab:blue')
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.savefig(output_path)
        plt.close()


def save_log(path, content):
    file = open(path, 'a')
    file.write(content)
    file.close()