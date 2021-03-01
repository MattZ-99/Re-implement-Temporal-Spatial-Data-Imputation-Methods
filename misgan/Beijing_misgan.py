from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import transforms

from dataset import data_preprocess
from dataset.Beijing_Taxi_dataset import Beijing_taxi_impute_dataset_without_timestamp
from nets.Beijing_critic import ConvCritic, FCCritic
from nets.Beijing_generator import (ConvDataGenerator, FCDataGenerator, ConvMaskGenerator, FCMaskGenerator)
from tools import utils, metric
from tools.utils import CriticUpdater
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

utils.setup_seed(210114)



# Discriminator
Critic = ConvCritic

# Generator
DataGenerator = ConvDataGenerator
MaskGenerator = ConvMaskGenerator

data_gen = DataGenerator().cuda()
mask_gen = MaskGenerator(hard_sigmoid=False).cuda()

data_critic = Critic().cuda()
mask_critic = Critic().cuda()

data_gen.train()
mask_gen.train()
data_critic.train()
mask_critic.train()

# parameter
batch_size = 16
nz = 128
lrate = 1e-4
gp_lambda = 1
tau = 0
epochs = 500
n_critic = 5
alpha = .2
missing_rate = 0.75
missing_type = "random"
# path
name = "Beijing_misgan_{}_mr_{}_mt_{}_geninter_{}".format(
    utils.time_for_save(), missing_rate, missing_type, n_critic)
model_path = "./outputs/{}/models/".format(name)
plot_path = "./outputs/{}/visualization/".format(name)
log_path = "./outputs/{}/logs/".format(name)
utils.makedirs(log_path)
utils.makedirs(model_path)
utils.makedirs(plot_path)


# dataset
train_set = Beijing_taxi_impute_dataset_without_timestamp(
    transforms=transforms.ToTensor(), missing_rate=missing_rate,
    missing_type=missing_type, train_proportion=0.95)

train_data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)

valid_set = Beijing_taxi_impute_dataset_without_timestamp(mode="valid",
    transforms=transforms.ToTensor(), missing_rate=missing_rate,
    missing_type=missing_type, train_proportion=0.95)
valid_data_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, drop_last=True)

data_noise = torch.FloatTensor(batch_size, nz).cuda()
mask_noise = torch.FloatTensor(batch_size, nz).cuda()

eps = torch.FloatTensor(batch_size, 1, 1, 1).cuda()
ones = torch.ones(batch_size).cuda()

# optimizer
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


data_critic_loss_curve = utils.ValuesVisual()
mask_critic_loss_curve = utils.ValuesVisual()
data_gen_loss_curve = utils.ValuesVisual()
mask_gen_loss_curve = utils.ValuesVisual()
distance_curve = utils.ValuesVisual()

for epoch in range(start_epoch, epochs):
    data_critic_loss = utils.ValueStat()
    mask_critic_loss = utils.ValueStat()
    data_gen_loss = utils.ValueStat()
    mask_gen_loss = utils.ValueStat()
    # train
    data_gen.train()
    mask_gen.train()
    data_critic.train()
    mask_critic.train()
    for real_data, real_mask in tqdm(train_data_loader, desc="epoch={}".format(epoch)):
        real_data = real_data.float()
        real_mask = real_mask.float()
        real_data = real_data.cuda()
        real_mask = real_mask.cuda()
        masked_real_data = utils.mask_data(real_data, real_mask, tau)

        data_noise.normal_()
        mask_noise.normal_()

        fake_data = data_gen(data_noise)
        fake_mask = mask_gen(mask_noise)

        masked_fake_data = utils.mask_data(fake_data, fake_mask, tau)

        update_data_critic(masked_real_data, masked_fake_data)
        update_mask_critic(real_mask, fake_mask)

        data_loss = update_data_critic.loss_value
        mask_loss = update_mask_critic.loss_value

        data_critic_loss.update(data_loss)
        mask_critic_loss.update(mask_loss)

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
            masked_fake_data = utils.mask_data(fake_data, fake_mask, tau)

            data_loss = -data_critic(masked_fake_data).mean()
            data_gen.zero_grad()
            data_loss.backward()
            data_gen_optimizer.step()

            data_gen_loss.update(data_loss)

            data_noise.normal_()
            mask_noise.normal_()

            fake_data = data_gen(data_noise)
            fake_mask = mask_gen(mask_noise)
            masked_fake_data = utils.mask_data(fake_data, fake_mask, tau)

            data_loss = -data_critic(masked_fake_data).mean()
            mask_loss = -mask_critic(fake_mask).mean()
            mask_loss_ = mask_loss + data_loss * alpha
            data_gen.zero_grad()
            mask_gen.zero_grad()
            mask_loss_.backward()
            mask_gen_optimizer.step()

            mask_gen_loss.update(mask_loss)

            for p in data_critic.parameters():
                p.requires_grad_(True)
            for p in mask_critic.parameters():
                p.requires_grad_(True)

    output = "epoch={}, data_critic_loss={}, mask_critic_loss={}, data_gen_loss={}, mask_gen_loss={}\n".format(
        epoch, data_critic_loss.get_avg(), mask_critic_loss.get_avg(), data_gen_loss.get_avg(), mask_gen_loss.get_avg()
    )
    print(output)
    utils.save_log(log_path + 'train' + '.log', output)

    data_critic_loss_curve.add_value(data_critic_loss.get_avg())
    mask_critic_loss_curve.add_value(mask_critic_loss.get_avg())
    data_gen_loss_curve.add_value(data_gen_loss.get_avg())
    mask_gen_loss_curve.add_value(mask_gen_loss.get_avg())


    # eval
    data_gen.eval()
    mask_gen.eval()
    data_critic.eval()
    mask_critic.eval()
    list_real_data = []
    list_fake_data = []
    with torch.no_grad():
        for real_data, real_mask in tqdm(valid_data_loader, desc="[Valid] epoch={}".format(epoch)):
            real_data = real_data.float()
            real_mask = real_mask.float()
            real_data = real_data.cuda()
            real_mask = real_mask.cuda()
            masked_real_data = utils.mask_data(real_data, real_mask, tau)

            data_noise.normal_()
            mask_noise.normal_()

            fake_data = data_gen(data_noise)
            fake_mask = mask_gen(mask_noise)
            masked_fake_data = utils.mask_data(fake_data, fake_mask, tau)

            masked_real_data = data_preprocess.scaler_for_impute(valid_set.scaler, masked_real_data)
            masked_fake_data = data_preprocess.scaler_for_impute(valid_set.scaler, masked_fake_data)
            list_real_data.append(masked_real_data)
            list_fake_data.append(masked_fake_data)

    all_real_data = torch.cat(list_real_data)
    all_fake_data = torch.cat(list_fake_data)
    first_wasserstein_distance = metric.compute_first_wasserstein_distance(all_real_data, all_fake_data)

    output = "[Valid] epoch={}, distance={}".format(epoch, first_wasserstein_distance)
    print(output)
    utils.save_log(log_path + 'valid' + '.log', output + '\n')
    distance_curve.add_value(first_wasserstein_distance)

    if epoch % 10 == 0:
        state = {'epoch': epoch,
                 'data_critic': data_critic.state_dict(),
                 'mask_critic': mask_critic.state_dict(),
                 'data_gen': data_gen.state_dict(),
                 'mask_gen': mask_gen.state_dict()
                 }
        torch.save(state, os.path.join(model_path, "epoch_{}_lr_{}.pth".format(
            epoch, data_critic_optimizer.param_groups[0]['lr']
                )
            )
        )
        data_critic_loss_curve.plot(plot_path + "data_critic_loss" + '.jpg')
        mask_critic_loss_curve.plot(plot_path + "mask_critic_loss" + '.jpg')
        data_gen_loss_curve.plot(plot_path + "data_gen_loss" + '.jpg')
        mask_gen_loss_curve.plot(plot_path + "mask_gen_loss" + '.jpg')
        distance_curve.plot(plot_path + 'Partial W-D' + '.jpg')