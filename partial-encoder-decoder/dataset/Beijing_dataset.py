from torch.utils.data import Dataset
import torch

import numpy as np
from sklearn import preprocessing


def load_train_data(dir, mode, train_proportion):
    data = np.load(dir)[:, :, :, 0]
    data_len = data.shape[0]
    split_point = int(data_len * train_proportion)
    if mode == "train":
        return data[0:split_point]
    elif mode == "valid":
        return data[split_point:data_len]
    elif mode == "test":
        return data[split_point:data_len]
    return data


def generate_imputation_mask(self):
    self.mask = torch.stack(tuple([self.generate_mask_sample(self) for _ in range(len(self))]), dim=0)


def generate_random_mask(self):
    random_mask = torch.raBeijing_TaxiFlow_dataset.pynd(self.len_time_stamp, self.grid_size_x, self.grid_size_y)
    random_mask[random_mask < self.missing_rate] = 0
    random_mask[random_mask >= self.missing_rate] = 1
    return random_mask


def generate_spatial_mask(self):
    spatial_mask = torch.ones(self.len_time_stamp, self.grid_size_x, self.grid_size_y)
    missing_length_x = int(self.grid_size_x * (self.missing_rate ** (1 / 2)))
    missing_length_y = int(self.grid_size_y * (self.missing_rate ** (1 / 2)))
    start_index_x = np.random.randint(0, self.grid_size_x - missing_length_x + 1)
    start_index_y = np.random.randint(0, self.grid_size_y - missing_length_y + 1)
    spatial_mask[:, start_index_x:start_index_x + missing_length_x, start_index_y:start_index_y + missing_length_y] \
        .fill_(0)
    return spatial_mask


def generate_temporal_mask(self):
    temporal_mask = torch.ones(self.len_time_stamp, self.grid_size_x, self.grid_size_y)
    missing_length = int(self.len_time_stamp * self.missing_rate)
    start_index = np.random.randint(0, self.len_time_stamp - missing_length + 1)
    temporal_mask[start_index:start_index + missing_length, :, :].fill_(0)
    return temporal_mask


def generate_spatial_temporal_mask(self):
    spatial_temporal_mask = torch.ones(self.len_time_stamp, self.grid_size_x, self.grid_size_y)
    missing_length_t = int(self.len_time_stamp * (self.missing_rate ** (1 / 3)))
    missing_length_x = int(self.grid_size_x * (self.missing_rate ** (1 / 3)))
    missing_length_y = int(self.grid_size_y * (self.missing_rate ** (1 / 3)))
    start_index_t = np.random.randint(0, self.len_time_stamp - missing_length_t + 1)
    start_index_x = np.random.randint(0, self.grid_size_x - missing_length_x + 1)
    start_index_y = np.random.randint(0, self.grid_size_y - missing_length_y + 1)
    spatial_temporal_mask[start_index_t:start_index_t + missing_length_t,
    start_index_x:start_index_x + missing_length_x, start_index_y:start_index_y + missing_length_y].fill_(0)
    return spatial_temporal_mask


class BeijingTaxiFlowDataset(Dataset):
    def __init__(self, raw_data_dir="./dataset/Beijing_TaxiFlow_data.npy", mode="train",
                 train_proportion=0.95, missing_type="random", missing_rate=0.25, feature_channels=1,
                 len_time_stamp=8, transforms=None, **kwargs):
        super().__init__()

        # load raw data
        data = load_train_data(raw_data_dir, mode, train_proportion)

        L, H, W = data.shape
        # data normalization using MinMaxScaler()
        # and scaler will be used to inverse transform
        data = data.reshape(L, H * W)
        scaler = preprocessing.MinMaxScaler()
        data = scaler.fit_transform(data).reshape(L, H, W)
        self.scaler = scaler
        self.data = data.astype(np.float32)
        self.data = torch.from_numpy(self.data)
        del data

        self.raw_data_len = L
        self.grid_size_x = H
        self.grid_size_y = W
        self.len_time_stamp = len_time_stamp

        self.missing_type = missing_type
        self.missing_rate = missing_rate
        self.transforms = transforms
        if self.missing_type == 'random':
            self.generate_mask_sample = generate_random_mask
        elif self.missing_type == 'spatial':
            self.generate_mask_sample = generate_spatial_mask
        elif self.missing_type == 'temporal':
            self.generate_mask_sample = generate_temporal_mask
        elif self.missing_type == 'spatial-temporal':
            self.generate_mask_sample = generate_spatial_temporal_mask
        else:
            raise NotImplementedError
        generate_imputation_mask(self)

    def __len__(self):
        return max(self.raw_data_len - self.len_time_stamp + 1, 0)

    def __getitem__(self, index):
        data_sample = self.data[index:index + self.len_time_stamp, :, :]
        mask_sample = self.mask[index]
        return data_sample, mask_sample
