# -*- coding=utf-8 -*-


import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn import preprocessing


# torch.manual_seed(0)
# torch.cuda.manual_seed_all(0)


def load_train_data(self):
    raw_data = np.load(self.raw_data_dir)
    self.num_data_timestamp = int(self.num_timestamp * self.train_proportion)
    self.data = torch.from_numpy(raw_data[:self.num_data_timestamp, :, :, :1]).float()


def load_test_data(self):
    raw_data = np.load(self.raw_data_dir)
    self.num_data_timestamp = self.num_timestamp - int(self.num_timestamp * self.train_proportion)
    self.data = torch.from_numpy(raw_data[-self.num_data_timestamp:, :, :, :1]).float()


def generate_imputation_mask(self):
    self.mask = torch.stack(tuple([self.generate_mask_sample(self) for _ in range(len(self))]), dim=0)


def generate_prediction_mask(self):
    self.mask = torch.stack(tuple(
        [torch.cat((self.generate_mask_sample(self), self.generate_mask_sample(self)), dim=0) for _ in
         range(len(self))]), dim=0)


def generate_prediction_deltas(self):
    self.deltas_forward = torch.stack(
        tuple([
            torch.cat(
                (self.generate_deltas_sample(self, self.mask[i].numpy()),
                 self.generate_deltas_sample(self, self.mask[i].numpy())), dim=0)
            for i in range(len(self))]), dim=0).float()
    self.deltas_backward = torch.stack(
        tuple([
            torch.cat(
                (self.generate_deltas_sample(self, self.mask[i].numpy()[::-1]),
                 self.generate_deltas_sample(self, self.mask[i].numpy()[::-1])), dim=0)
            for i in range(len(self))]), dim=0).float()


def generate_prediction_forwards(self):
    self.forwards_f = torch.stack(
        tuple([
            torch.cat((self.generate_forwards_sample(self, self.data[i:i + self.seq_len, :].numpy(),
                                                     self.mask[i].numpy()),
                       self.generate_forwards_sample(self, self.data[i:i + self.seq_len, :].numpy(),
                                                     self.mask[i].numpy()))
                      , dim=0)
            for i in range(len(self))]), dim=0).float()
    self.forwards_b = torch.stack(
        tuple([
            torch.cat((self.generate_forwards_sample(self, self.data[i:i + self.seq_len, :].numpy()[::-1],
                                                     self.mask[i].numpy()[::-1]),
                       self.generate_forwards_sample(self, self.data[i:i + self.seq_len, :].numpy()[::-1],
                                                     self.mask[i].numpy()[::-1]))
                      , dim=0)
            for i in range(len(self))]), dim=0).float()


def get_imputation_item(self, index):
    data_forward_sample = self.data[index:index + self.seq_len, :]
    data_backward_sample = data_forward_sample.clone()
    data_backward_sample = data_backward_sample.numpy()[::-1].copy()
    data_backward_sample = torch.from_numpy(data_backward_sample)

    mask_forward_sample = self.mask[index]
    mask_backward_sample = mask_forward_sample.clone().numpy()[::-1].copy()
    mask_backward_sample = torch.from_numpy(mask_backward_sample)

    deltas_forward_sample = self.deltas_forward[index]
    deltas_backward_sample = self.deltas_backward[index]
    forwards_forward_sample = self.forwards_f[index]
    forwards_backward_sample = self.forwards_b[index]
    return data_forward_sample, data_backward_sample, mask_forward_sample, mask_backward_sample, \
           deltas_forward_sample, deltas_backward_sample, forwards_forward_sample, forwards_backward_sample


def get_prediction_item(self, index):
    data_forward_sample = self.data[index:index + self.seq_len * 2, :]
    data_backward_sample = data_forward_sample.clone()
    data_backward_sample = data_backward_sample.numpy()[::-1].copy()
    data_backward_sample = torch.from_numpy(data_backward_sample)

    mask_forward_sample = self.mask[index]
    mask_backward_sample = mask_forward_sample.clone().numpy()[::-1].copy()
    mask_backward_sample = torch.from_numpy(mask_backward_sample)

    deltas_forward_sample = self.deltas_forward[index]
    deltas_backward_sample = self.deltas_backward[index]
    forwards_forward_sample = self.forwards_f[index]
    forwards_backward_sample = self.forwards_b[index]
    return data_forward_sample, data_backward_sample, mask_forward_sample, mask_backward_sample, \
           deltas_forward_sample, deltas_backward_sample, forwards_forward_sample, forwards_backward_sample


def get_imputation_length(self):
    return max(self.num_data_timestamp - self.seq_len + 1, 0)


def get_prediction_length(self):
    return max(self.num_data_timestamp - self.seq_len * 2 + 1, 0)


def generate_random_mask(self):
    random_mask = torch.rand(self.seq_len, self.grid_size_x, self.grid_size_y, self.feature_channels)
    random_mask[random_mask < self.missing_rate] = 0
    random_mask[random_mask >= self.missing_rate] = 1
    return random_mask


def generate_temporal_mask(self):
    temporal_mask = torch.ones(self.seq_len, self.grid_size_x, self.grid_size_y, self.feature_channels)
    missing_length = int(self.seq_len * self.missing_rate)
    start_index = np.random.randint(0, self.seq_len - missing_length + 1)
    temporal_mask[start_index:start_index + missing_length, :, :, :].fill_(0)
    return temporal_mask


def generate_spatial_mask(self):
    spatial_mask = torch.ones(self.seq_len, self.grid_size_x, self.grid_size_y, self.feature_channels)
    missing_length_x = int(self.grid_size_x * (self.missing_rate ** (1 / 2)))
    missing_length_y = int(self.grid_size_y * (self.missing_rate ** (1 / 2)))
    start_index_x = np.random.randint(0, self.grid_size_x - missing_length_x + 1)
    start_index_y = np.random.randint(0, self.grid_size_y - missing_length_y + 1)
    spatial_mask[:, start_index_x:start_index_x + missing_length_x, start_index_y:start_index_y + missing_length_y,
    :].fill_(0)
    return spatial_mask


def generate_spatial_temporal_mask(self):
    spatial_temporal_mask = torch.ones(self.seq_len, self.grid_size_x, self.grid_size_y, self.feature_channels)
    missing_length_t = int(self.seq_len * (self.missing_rate ** (1 / 3)))
    missing_length_x = int(self.grid_size_x * (self.missing_rate ** (1 / 3)))
    missing_length_y = int(self.grid_size_y * (self.missing_rate ** (1 / 3)))
    start_index_t = np.random.randint(0, self.seq_len - missing_length_t + 1)
    start_index_x = np.random.randint(0, self.grid_size_x - missing_length_x + 1)
    start_index_y = np.random.randint(0, self.grid_size_y - missing_length_y + 1)
    spatial_temporal_mask[start_index_t:start_index_t + missing_length_t,
    start_index_x:start_index_x + missing_length_x, start_index_y:start_index_y + missing_length_y, :].fill_(0)
    return spatial_temporal_mask


def parse_delta_sample(self, mask):
    deltas = []

    for h in range(self.seq_len):
        if h == 0:
            deltas.append(np.ones(self.element_num))
        else:
            deltas.append(np.ones(self.element_num) + (1 - mask[h]) * deltas[-1])

    deltas = np.array(deltas)
    return torch.from_numpy(deltas)


def generate_imputation_deltas(self):
    self.deltas_forward = torch.stack(
        tuple([self.generate_deltas_sample(self, self.mask[i].numpy()) for i in range(len(self))]), dim=0).float()
    self.deltas_backward = torch.stack(
        tuple([self.generate_deltas_sample(self, self.mask[i].numpy()[::-1]) for i in range(len(self))]), dim=0).float()


def gen_forwards_sample(self, value, mask):
    value = np.where(mask, value, np.nan)
    forwards = pd.DataFrame(value).fillna(method='ffill').fillna(0.0).values
    return torch.from_numpy(forwards)


def generate_imputation_forwards(self):
    self.forwards_f = torch.stack(
        tuple([self.generate_forwards_sample(self, self.data[i:i + self.seq_len, :].numpy(), self.mask[i].numpy())
               for i in range(len(self))]), dim=0).float()
    self.forwards_b = torch.stack(
        tuple([self.generate_forwards_sample(self, self.data[i:i + self.seq_len, :].numpy()[::-1],
                                             self.mask[i].numpy()[::-1]) for i in range(len(self))]), dim=0).float()


class BeijingTaxiFlow(Dataset):
    def __init__(self, raw_data_dir, missing_type, missing_rate, num_timestamp, train_proportion, seq_len, grid_size_x,
                 grid_size_y, feature_channels):
        self.raw_data_dir = raw_data_dir
        self.missing_type = missing_type
        self.missing_rate = missing_rate
        self.num_timestamp = num_timestamp
        self.train_proportion = train_proportion
        self.seq_len = seq_len
        self.grid_size_x = grid_size_x
        self.grid_size_y = grid_size_y
        self.feature_channels = feature_channels
        self.element_num = self.grid_size_x * self.grid_size_y * self.feature_channels
        self.load_data(self)
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
        self.generate_mask(self)

        self.data = self.data.view(self.num_data_timestamp, -1).contiguous()
        self.mask = self.mask.view(len(self), self.seq_len, -1).contiguous()

        self.data = self.data.numpy()
        scaler = preprocessing.MinMaxScaler()
        self.data = scaler.fit_transform(self.data)
        self.scaler = scaler
        self.data = torch.from_numpy(self.data)

        self.generate_deltas_sample = parse_delta_sample
        self.generate_deltas(self)
        self.generate_forwards_sample = gen_forwards_sample
        self.generate_forwards(self)

    def load_data(self):
        raise NotImplementedError

    def generate_mask(self):
        raise NotImplementedError

    def generate_deltas(self):
        raise NotImplementedError

    def generate_forwards(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class BeijingTaxiFlowImputationTrainSet(BeijingTaxiFlow):
    def __init__(self, raw_data_dir, missing_type, missing_rate, num_timestamp, train_proportion, seq_len, grid_size_x,
                 grid_size_y, feature_channels):
        self.load_data = load_train_data
        self.generate_mask = generate_imputation_mask
        self.generate_deltas = generate_imputation_deltas
        self.generate_forwards = generate_imputation_forwards
        super().__init__(raw_data_dir, missing_type, missing_rate, num_timestamp, train_proportion, seq_len,
                         grid_size_x, grid_size_y, feature_channels)

    __getitem__ = get_imputation_item
    __len__ = get_imputation_length


class BeijingTaxiFlowImputationTestSet(BeijingTaxiFlow):
    def __init__(self, raw_data_dir, missing_type, missing_rate, num_timestamp, train_proportion, seq_len, grid_size_x,
                 grid_size_y, feature_channels):
        self.load_data = load_test_data
        self.generate_mask = generate_imputation_mask
        self.generate_deltas = generate_imputation_deltas
        self.generate_forwards = generate_imputation_forwards
        super().__init__(raw_data_dir, missing_type, missing_rate, num_timestamp, train_proportion, seq_len,
                         grid_size_x, grid_size_y, feature_channels)

    __getitem__ = get_imputation_item
    __len__ = get_imputation_length


class BeijingTaxiFlowPredictionTrainSet(BeijingTaxiFlow):
    def __init__(self, raw_data_dir, missing_type, missing_rate, num_timestamp, train_proportion, seq_len, grid_size_x,
                 grid_size_y, feature_channels):
        self.load_data = load_train_data
        self.generate_mask = generate_prediction_mask
        self.generate_deltas = generate_prediction_deltas
        self.generate_forwards = generate_prediction_forwards
        super().__init__(raw_data_dir, missing_type, missing_rate, num_timestamp, train_proportion, seq_len,
                         grid_size_x, grid_size_y, feature_channels)

    __getitem__ = get_prediction_item
    __len__ = get_prediction_length


class BeijingTaxiFlowPredictionTestSet(BeijingTaxiFlow):
    def __init__(self, raw_data_dir, missing_type, missing_rate, num_timestamp, train_proportion, seq_len, grid_size_x,
                 grid_size_y, feature_channels):
        self.load_data = load_test_data
        self.generate_mask = generate_prediction_mask
        self.generate_deltas = generate_prediction_deltas
        self.generate_forwards = generate_prediction_forwards
        super().__init__(raw_data_dir, missing_type, missing_rate, num_timestamp, train_proportion, seq_len,
                         grid_size_x, grid_size_y, feature_channels)

    __getitem__ = get_prediction_item
    __len__ = get_prediction_length


def load_synthetic_train_data(self):
    raw_data = torch.load(self.raw_data_dir + 'data.pt', map_location=torch.device('cpu'))
    self.num_data_timestamp = int(self.num_timestamp * self.train_proportion)
    self.data = raw_data[:self.num_data_timestamp]


def load_synthetic_test_data(self):
    raw_data = torch.load(self.raw_data_dir + 'data.pt', map_location=torch.device('cpu'))
    self.num_data_timestamp = self.num_timestamp - int(self.num_timestamp * self.train_proportion)
    self.data = raw_data[-self.num_data_timestamp:]


def load_synthetic_train_mask(self):
    raw_mask = torch.load(self.raw_data_dir + 'mask.pt', map_location=torch.device('cpu'))
    self.num_data_timestamp = int(self.num_timestamp * self.train_proportion)
    self.mask = raw_mask[:self.num_data_timestamp]


def load_synthetic_test_mask(self):
    raw_mask = torch.load(self.raw_data_dir + 'mask.pt', map_location=torch.device('cpu'))
    self.num_data_timestamp = self.num_timestamp - int(self.num_timestamp * self.train_proportion)
    self.mask = raw_mask[-self.num_data_timestamp:]


class BeijingTaxiFlowSyntheticTrainDataset(BeijingTaxiFlow):
    def __init__(self, raw_data_dir, missing_type, missing_rate, num_timestamp, train_proportion, seq_len, grid_size_x,
                 grid_size_y, feature_channels):
        self.load_data = load_synthetic_train_data
        self.generate_mask = load_synthetic_train_mask
        super().__init__(raw_data_dir, missing_type, missing_rate, num_timestamp, train_proportion, seq_len,
                         grid_size_x, grid_size_y, feature_channels)

    def __getitem__(self, index):
        return self.data[index], self.mask[index]

    def __len__(self):
        return self.data.size(0)


class BeijingTaxiFlowSyntheticTestDataset(BeijingTaxiFlow):
    def __init__(self, raw_data_dir, missing_type, missing_rate, num_timestamp, train_proportion, seq_len, grid_size_x,
                 grid_size_y, feature_channels):
        self.load_data = load_synthetic_test_data
        self.generate_mask = load_synthetic_test_mask
        super().__init__(raw_data_dir, missing_type, missing_rate, num_timestamp, train_proportion, seq_len,
                         grid_size_x, grid_size_y, feature_channels)

    def __getitem__(self, index):
        return self.data[index], self.mask[index]

    def __len__(self):
        return self.data.size(0)
