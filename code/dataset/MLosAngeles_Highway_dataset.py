from torch import nn
import numpy as np
import torch
from torchvision import transforms


def load_train_data(dir, mode, train_proportion):
    data = np.load(dir)[:, :, 0]
    data_len = data.shape[0]
    split_point = int(data_len * train_proportion)
    if mode == "train":
        return data[0:split_point]
    elif mode == "valid":
        return data[split_point:data_len]
    elif mode == "test":
        return data[split_point:data_len]


def generate_random_mask_without_timestamp(data_len, data_size, missing_rate, feature_channels=1):
    random_mask = torch.rand(data_len, feature_channels, data_size)
    random_mask[random_mask < missing_rate] = 0
    random_mask[random_mask >= missing_rate] = 1
    return random_mask


class LosAngeles_impute_dataset_without_timestamp(Dataset):
    def __init__(self, raw_data_dir="dataset/LosAngeles_HighwaySpeed_data.npy", mode="train", train_proportion=0.7,
                 missing_type="random", missing_rate=0.2):
        super().__init__()
        self.data = load_train_data(raw_data_dir, mode, train_proportion)

        self.data_len, self.data_size = self.data.shape

        self.missing_type = missing_type
        self.missing_rate = missing_rate

        if self.missing_type == "random":
            self.mask = generate_random_mask_without_timestamp(self.data_len, self.data_size, self.missing_rate)

    def __getitem__(self, item):
        output_data = transforms.ToTensor()(self.data[item])
        output_mask = self.mask[item]
        return output_data, output_mask

    def __len__(self):
        return self.data_len
