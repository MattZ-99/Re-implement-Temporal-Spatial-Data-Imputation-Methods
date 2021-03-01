from time import sleep
from tqdm import tqdm

import time

print()

# from tqdm import tqdm
# from collections import OrderedDict
#
# total = 10000  # 总迭代次数
# loss = total
# with tqdm(total=total, desc="进度条") as pbar:
#     for i in range(total):
#         loss -= 1
#         #        pbar.set_postfix(OrderedDict(loss='{0:1.5f}'.format(loss)))
#         pbar.set_postfix({'loss': '{0:1.5f}'.format(loss)})  # 输入一个字典，显示实验指标
#         pbar.update(1)

# def blabla():
#   tqdm.write("Foo blabla")
# i = 0
# pbar = tqdm(range(3), postfix=str(i), desc=str(i))
# for k in pbar:
#     i = k
#   # blabla()
#     sleep(.5)
#     pbar.set_postfix(str(i))


# data_critic_optimizer = optim.Adam(
#     data_critic.parameters(), lr=lrate, betas=(.5, .9))

# import torch
# import torch.optim as optim
# import torch.nn as nn
# layer = nn.Linear(1,1)
# o = optim.SGD(layer.parameters(), lr=0.1)
#
# print(o.param_groups[0]['lr'])


# import numpy as np
# from sklearn import preprocessing
#
# data = np.load("dataset/Beijing_TaxiFlow_data.npy")[:, :, :, 0]
#
# L, H, W = data.shape
#
# data = data.reshape(L, H * W)
#
# scaler = preprocessing.MinMaxScaler()
# data = scaler.fit_transform(data)
#
# print(len(scaler.data_max_))
#
# import torch
#
# lmj = torch.Tensor(data)

# import numpy as np
# import torchvision.transforms as transforms
# import torch
#
# data = np.load("dataset/Beijing_TaxiFlow_data.npy")[:, :, :, 0]
#
# for i in data:
#     i = i.astype(np.uint8)
#     i = transforms.ToTensor()(i)
#     print(i.shape, torch.max(i), torch.min(i))

import numpy as np
# data = np.load("dataset/LosAngeles_HighwaySpeed_data.npy")
#
# for i in data:
#     print(np.max(i), np.average(i), np.min(i))

# data = np.load("dataset/Beijing_TaxiFlow_data.npy")[:, :, :, 0]  # 1285.0 0.0 103.85968420971399
# data = np.load("dataset/LosAngeles_HighwaySpeed_data.npy") # 70.0 0.0 53.71902110241347
# print(np.max(data), np.min(data), np.average(data))

# import torch
#
# x = torch.rand(32, 32)
# ln = torch.nn.Linear(10, 20)
# print(x.requires_grad)
#
# for i in ln.parameters():
#     print(i)


# import torch
# import torch.nn as nn
#
# DIM = 64
# latent_size = 128
#
# preprocess = nn.Sequential(
#     nn.Linear(latent_size, 4 * 5 * 5 * DIM),
#     nn.ReLU(True),
# )
# block1 = nn.Sequential(
#     nn.ConvTranspose2d(4 * DIM, 2 * DIM, 3, padding=1),
#     nn.ReLU(True),
# )
# block2 = nn.Sequential(
#     nn.ConvTranspose2d(2 * DIM, DIM, 3),
#     nn.ReLU(True),
# )
# deconv_out = nn.ConvTranspose2d(DIM, 1, 4, stride=2)
#
#
# data = torch.FloatTensor(64, 128, 1, 1)
# data = block2(data)
# data = deconv_out(data)
#
# data = torch.FloatTensor(64, 128)
# print(data.shape)
# data = preprocess(data)
# print(data.shape)
# data = data.view(-1, 4 * DIM, 5, 5)
# print(data.shape)
# data = block1(data)
# print(data.shape)
# # data = data[:, :, :5, :5]
# data = block2(data)
# print(data.shape)
# data = deconv_out(data)
# print(data.shape)
# data = (lambda x: torch.sigmoid(x).view(-1, 1, 256))(data)
# data = nn.Linear(256, 207)(data)
# print(data.shape)


# from nets.mnist_generator import (ConvDataGenerator, FCDataGenerator, ConvMaskGenerator, FCMaskGenerator)
# import torch
# import torch.nn as nn
# DataGenerator = ConvDataGenerator
# MaskGenerator = ConvMaskGenerator
# data_gen = DataGenerator().cuda()
#
# data_gen = DataGenerator().cuda()
# mask_gen = MaskGenerator(hard_sigmoid=False).cuda()
#
# data_noise = torch.FloatTensor(64, 128).cuda()
# mask_noise = torch.FloatTensor(64, 128).cuda()
#
# data_noise.normal_()
# mask_noise.normal_()
#
# fake_data = data_gen(data_noise)
# fake_mask = mask_gen(mask_noise)
#
# print(fake_data.shape, fake_mask.shape)


# DIM = 64
# preprocess = nn.Sequential(
#             nn.Linear(128, 4 * 5 * 5 * DIM),
#             nn.ReLU(True),
#         )
# block1 = nn.Sequential(
#     nn.ConvTranspose2d(4 * DIM, 2 * DIM, 5),
#     nn.ReLU(True),
# )
# block2 = nn.Sequential(
#     nn.ConvTranspose2d(2 * DIM, DIM, 5),
#     nn.ReLU(True),
# )
# deconv_out = nn.ConvTranspose2d(DIM, 1, 8, stride=2)
# data = torch.FloatTensor(128)
#
# net = preprocess(data)
# print(net.shape)
# net = net.view(-1, 4 * DIM, 5, 5)
# print(net.shape)
# net = block1(net)
# print(net.shape)
# net = net[:, :, :9, :9]
# print(net.shape)
# net = block2(net)
# print(net.shape)
# net = deconv_out(net)
# print(net.shape)

# import torch
# import torch.nn as nn
#
# DIM = 64
# main = nn.Sequential(
#     nn.Conv2d(1, DIM, 5, stride=2, padding=2),
#     nn.ReLU(True),
#     nn.Conv2d(DIM, 2 * DIM, 5, stride=2, padding=2),
#     nn.ReLU(True),
#     nn.Conv2d(2 * DIM, 4 * DIM, 5, stride=2, padding=2),
#     nn.ReLU(True),
# )
#
# output = nn.Linear(4 * 4 * 4 * DIM, 1)
#
# data = torch.FloatTensor(64, 1, 32, 32)
# print(data.shape)
# input = data.view(-1, 1, 32, 32)
# print(input.shape)
# net = main(input)
# print(net.shape)
# net = net.view(-1, 4 * 4 * 4 * DIM)
# print(net.shape)