from nets.mnist_critic import ConvCritic
from nets.LosAngeles_generator import ConvDataGenerator, ConvMaskGenerator
from torch.utils.data import DataLoader
from dataset.MLosAngeles_Highway_dataset import LosAngeles_impute_dataset_without_timestamp
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

DataGenerator = ConvDataGenerator
MaskGenerator = ConvMaskGenerator

Critic = ConvCritic

train_set = LosAngeles_impute_dataset_without_timestamp()

batch_size = 1

data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)

for i, (real_data, real_mask) in enumerate(data_loader):
    print(real_data.shape, real_mask.shape)