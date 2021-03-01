import torch.nn as nn
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset
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


def generate_random_mask(seq_len, grid_size_x, grid_size_y, missing_rate, feature_channels=1):
    random_mask = torch.rand(seq_len, feature_channels, grid_size_x, grid_size_y)
    random_mask[random_mask < missing_rate] = 0
    random_mask[random_mask >= missing_rate] = 1
    return random_mask


class Beijing_taxi_impute_dataset(Dataset):
    # def __init__(self, **kwargs):
    #     kwargs.setdefault("raw_data_dir", "./dataset/Beijing_TaxiFlow_data.npy")
    #     kwargs.setdefault("missing_type", "random")
    #     kwargs.setdefault("missing_rate", 0.2)
    #     kwargs.setdefault("seq_len", 8)
    #     kwargs.setdefault("feature_channels", 1)
    #     pass
    def __init__(self, raw_data_dir="dataset/Beijing_TaxiFlow_data.npy", mode="train", train_proportion=0.7,
                 missing_type="random", missing_rate=0.2, seq_len=8, feature_channels=1):
        super().__init__()
        self.data = load_train_data(raw_data_dir, mode, train_proportion)
        _, grid_size_x, grid_size_y = self.data.shape
        self.seq_len = seq_len
        self.grid_size_x = grid_size_x
        self.grid_size_y = grid_size_y
        self.missing_type = missing_type
        self.miss_rate = missing_rate

    def __getitem__(self, item):
        if self.missing_type == "random":
            self.mask = generate_random_mask(self.seq_len, self.grid_size_x,
                                             self.grid_size_y, self.missing_rate, self.feature_channels)

    def __len__(self):
        pass


def generate_random_mask_without_timestamp(data_len, grid_size_x, grid_size_y, missing_rate, feature_channels=1):
    random_mask = torch.rand(data_len, feature_channels, grid_size_x, grid_size_y)
    random_mask[random_mask < missing_rate] = 0
    random_mask[random_mask >= missing_rate] = 1
    return random_mask


class Beijing_taxi_impute_dataset_without_timestamp(Dataset):
    def __init__(self, raw_data_dir="dataset/Beijing_TaxiFlow_data.npy", mode="train", train_proportion=0.7,
                 missing_type="random", missing_rate=0.2, feature_channels=1, transforms=None):
        super().__init__()

        data = load_train_data(raw_data_dir, mode, train_proportion)

        L, H, W = data.shape
        data = data.reshape(L, H * W)
        scaler = preprocessing.MinMaxScaler()
        data = scaler.fit_transform(data).reshape(L, H, W)
        self.scaler = scaler
        self.data = data.astype(np.float32)

        self.data_len, grid_size_x, grid_size_y = self.data.shape
        self.grid_size_x = grid_size_x
        self.grid_size_y = grid_size_y
        self.missing_type = missing_type
        self.missing_rate = missing_rate
        self.feature_channels = feature_channels
        self.transforms = transforms

        if self.missing_type == "random":
            self.mask = generate_random_mask_without_timestamp(self.data_len, self.grid_size_x, self.grid_size_y,
                                                               self.missing_rate, self.feature_channels)

    def __getitem__(self, item):
        output_data = self.data[item]
        if self.transforms:
            output_data = self.transforms(output_data)
        output_mask = self.mask[item]
        return output_data, output_mask

    def __len__(self):
        return self.data_len
