from torch.autograd import grad
import torch
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import time


class CriticUpdater:
    def __init__(self, critic, critic_optimizer, eps, ones, gp_lambda=10):
        self.critic = critic
        self.critic_optimizer = critic_optimizer
        self.eps = eps
        self.ones = ones
        self.gp_lambda = gp_lambda

    def __call__(self, real, fake):
        real = real.detach()
        fake = fake.detach()
        self.critic.zero_grad()
        self.eps.uniform_(0, 1)

        interp = (self.eps * real + (1 - self.eps) * fake).requires_grad_()
        grad_d = grad(self.critic(interp), interp, grad_outputs=self.ones,
                      create_graph=True)[0]
        grad_d = grad_d.view(real.shape[0], -1)
        grad_penalty = ((grad_d.norm(dim=1) - 1)**2).mean() * self.gp_lambda
        w_dist = self.critic(fake).mean() - self.critic(real).mean()
        loss = w_dist + grad_penalty
        loss.backward()
        self.critic_optimizer.step()
        self.loss_value = loss.item()


def mask_norm(diff, mask):
    """Mask normalization"""
    dim = 1, 2, 3
    # Assume mask.sum(1) is non-zero throughout
    return ((diff * mask).sum(dim) / mask.sum(dim)).mean()


def mkdir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def mask_data(data, mask, tau):
    return mask * data + (1 - mask) * tau


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


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


def save_log(path, content):
    file = open(path, 'a')
    file.write(content)
    file.close()


def makedirs(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


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


def time_for_save():
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())
