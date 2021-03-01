from dataset.Beijing_Taxi_dataset import Beijing_taxi_impute_dataset_without_timestamp
from nets.mnist_generator import (ConvDataGenerator, FCDataGenerator, ConvMaskGenerator, FCMaskGenerator)
from nets.mnist_critic import ConvCritic, FCCritic
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.optim as optim
import numpy as np

from tools.utils import CriticUpdater, mask_data
import time

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

DataGenerator = ConvDataGenerator
MaskGenerator = ConvMaskGenerator

Critic = ConvCritic

train_set = Beijing_taxi_impute_dataset_without_timestamp()

data_gen = DataGenerator().cuda()
mask_gen = MaskGenerator(hard_sigmoid=False).cuda()

data_critic = Critic().cuda()
mask_critic = Critic().cuda()
n_critic = 5
gp_lambda = 1
batch_size = 64

nz = 128
epochs = 500
plot_interval = 100
save_interval = 100
alpha = .2
tau = 0

data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)

data_noise = torch.FloatTensor(batch_size, nz).cuda()
mask_noise = torch.FloatTensor(batch_size, nz).cuda()
eps = torch.FloatTensor(batch_size, 1, 1, 1).cuda()
ones = torch.ones(batch_size).cuda()

lrate = 1e-4

data_gen_optimizer = optim.Adam(
    data_gen.parameters(), lr=lrate, betas=(.5, .9))
mask_gen_optimizer = optim.Adam(
    mask_gen.parameters(), lr=lrate, betas=(.5, .9))

data_critic_optimizer = optim.Adam(
    data_critic.parameters(), lr=lrate, betas=(.5, .9))
mask_critic_optimizer = optim.Adam(
    mask_critic.parameters(), lr=lrate, betas=(.5, .9))

update_data_critic = CriticUpdater(
    data_critic, data_critic_optimizer, eps, ones, gp_lambda)
update_mask_critic = CriticUpdater(
    mask_critic, mask_critic_optimizer, eps, ones, gp_lambda)

start_epoch = 0
critic_updates = 0
n_batch = len(data_loader)
start = time.time()
epoch_start = start

for epoch in range(start_epoch, epochs):
    sum_data_loss, sum_mask_loss = 0, 0
    for real_data, real_mask in tqdm(data_loader, desc="epoch={}".format(epoch)):
        # Assume real_data and mask have the same number of channels.
        # Could be modified to handle multi-channel images and
        # single-channel masks.
        real_data = real_data.float()
        real_mask = real_mask.float()
        real_data = real_data.cuda()
        real_mask = real_mask.cuda()

        masked_real_data = mask_data(real_data, real_mask, tau)

        data_noise.normal_()
        mask_noise.normal_()

        fake_data = data_gen(data_noise)
        fake_mask = mask_gen(mask_noise)

        masked_fake_data = mask_data(fake_data, fake_mask, tau)

        update_data_critic(masked_real_data, masked_fake_data)
        update_mask_critic(real_mask, fake_mask)

        sum_data_loss += update_data_critic.loss_value
        sum_mask_loss += update_mask_critic.loss_value

        critic_updates += 1

        if critic_updates == n_critic:
            critic_updates = 0

            for p in data_critic.parameters():
                p.requires_grad_(False)
            for p in mask_critic.parameters():
                p.requires_grad_(False)

            data_noise.normal_()
            mask_noise.normal_()

            fake_data = data_gen(data_noise)
            fake_mask = mask_gen(mask_noise)
            masked_fake_data = mask_data(fake_data, fake_mask, tau)

            data_loss = -data_critic(masked_fake_data).mean()
            data_gen.zero_grad()
            data_loss.backward()
            data_gen_optimizer.step()

            data_noise.normal_()
            mask_noise.normal_()

            fake_data = data_gen(data_noise)
            fake_mask = mask_gen(mask_noise)
            masked_fake_data = mask_data(fake_data, fake_mask, tau)

            data_loss = -data_critic(masked_fake_data).mean()
            mask_loss = -mask_critic(fake_mask).mean()
            data_gen.zero_grad()
            mask_gen.zero_grad()
            (mask_loss + data_loss * alpha).backward()
            mask_gen_optimizer.step()

            for p in data_critic.parameters():
                p.requires_grad_(True)
            for p in mask_critic.parameters():
                p.requires_grad_(True)

        mean_data_loss = sum_data_loss / n_batch
        mean_mask_loss = sum_mask_loss / n_batch

    epoch_end = time.time()
    time_elapsed = epoch_end - start
    epoch_time = epoch_end - epoch_start
    epoch_start = epoch_end
    with open('logs/time.txt', 'a') as f:
        print(epoch, epoch_time, time_elapsed, file=f)

