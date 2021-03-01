# -*- coding=utf-8 -*-


import numpy as np
import torch
from torch_geometric.data import Data, Batch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# torch.manual_seed(0)
# torch.cuda.manual_seed_all(0)


def _generate_random_raw_mask_sample(self):
    raw_mask_sample = torch.rand(self.seq_len, self.graph_size, self.feature_channels)
    raw_mask_sample[raw_mask_sample < self.missing_rate] = 0
    raw_mask_sample[raw_mask_sample >= self.missing_rate] = 1
    return raw_mask_sample


def _generate_spatial_raw_mask_sample(self):
    raw_mask_sample = torch.ones(self.seq_len, self.graph_size, self.feature_channels)
    num_missing_node = int(self.graph_size * self.missing_rate)
    selected_node = self._random_walk(num_missing_node)
    raw_mask_sample[:, selected_node, :] = 0
    return raw_mask_sample


def _generate_temporal_raw_mask_sample(self):
    raw_mask_sample = torch.ones(self.seq_len, self.graph_size, self.feature_channels)
    missing_length = int(self.seq_len * self.missing_rate)
    start_index = np.random.randint(0, self.seq_len - missing_length + 1)
    # start_index = torch.randint(0, self.seq_len - missing_length + 1).item()
    raw_mask_sample[start_index: start_index + missing_length, :, :] = 0
    return raw_mask_sample


def _generate_spatial_temporal_raw_mask_sample(self):
    raw_mask_sample = torch.ones(self.seq_len, self.graph_size, self.feature_channels)
    num_missing_node = int(self.graph_size * (self.missing_rate ** (1 / 2)))
    selected_node = self._random_walk(num_missing_node)
    missing_length = int(self.seq_len * (self.missing_rate ** (1 / 2)))
    start_index = np.random.randint(0, self.seq_len - missing_length + 1)
    # start_index = torch.randint(0, self.seq_len - missing_length + 1).item()
    raw_mask_sample[start_index: start_index + missing_length, selected_node, :] = 0
    return raw_mask_sample


def _load_train_raw_data(self):
    raw_data = np.load(self.raw_data_dir)
    self.num_timestamp = int(self.num_timestamp * self.train_proportion)
    raw_data = raw_data[:self.num_timestamp, :, :].reshape(self.num_timestamp, self.graph_size * self.feature_channels)
    self.scaler = MinMaxScaler()
    self.scaler.fit(raw_data)
    raw_data = torch.from_numpy(
        self.scaler.transform(raw_data).reshape(self.num_timestamp, self.graph_size, self.feature_channels)).float()
    return raw_data


def _load_test_raw_data(self):
    raw_data = np.load(self.raw_data_dir)
    self.num_timestamp = self.num_timestamp - int(self.num_timestamp * self.train_proportion)
    raw_data = raw_data[-self.num_timestamp:, :, :].reshape(self.num_timestamp, self.graph_size * self.feature_channels)
    self.scaler = MinMaxScaler()
    self.scaler.fit(raw_data)
    raw_data = torch.from_numpy(
        self.scaler.transform(raw_data).reshape(self.num_timestamp, self.graph_size, self.feature_channels)).float()
    return raw_data


def _generate_imputation_raw_mask(self):
    raw_mask = torch.stack(tuple([self.generate_raw_mask_sample(self) for _ in range(len(self))]), dim=0)
    return raw_mask


def _generate_prediction_raw_mask(self):
    raw_mask = torch.stack(tuple(
        [torch.cat((self.generate_mask_sample(self), self.generate_mask_sample(self)), dim=0) for _ in
         range(len(self))]), dim=0)
    return raw_mask


def _build_imputation_data(self, raw_data):
    raw_data = [Data(x=raw_data[t], edge_index=self.edge_index, edge_weight=self.edge_weight) for t in
                range(self.num_timestamp)]
    self.data = []
    for t in range(len(self)):
        temp_batch = Batch()
        temp_batch = temp_batch.from_data_list(raw_data[t: t + self.seq_len])
        data_sample = Data(x=temp_batch.x, edge_index=temp_batch.edge_index, edge_weight=temp_batch.edge_weight)
        self.data.append(data_sample)


def _build_prediction_data(self, raw_data):
    raw_data = [Data(x=raw_data[t], edge_index=self.edge_index, edge_weight=self.edge_weight) for t in
                range(self.num_timestamp)]
    self.past_data, self.future_data = [], []
    for t in range(len(self)):
        temp_batch = Batch()
        temp_batch = temp_batch.from_data_list(raw_data[t: t + self.seq_len])
        data_sample = Data(x=temp_batch.x, edge_index=temp_batch.edge_index, edge_weight=temp_batch.edge_weight)
        self.past_data.append(data_sample)
        temp_batch = temp_batch.from_data_list(raw_data[t + self.seq_len: t + self.seq_len * 2])
        data_sample = Data(x=temp_batch.x, edge_index=temp_batch.edge_index, edge_weight=temp_batch.edge_weight)
        self.future_data.append(data_sample)


def _build_imputation_mask(self, raw_mask):
    self.mask = []
    for i in range(len(self)):
        raw_mask_sample = [Data(x=raw_mask[i][t], edge_index=self.edge_index, edge_weight=self.edge_weight) for t in
                           range(self.seq_len)]
        temp_batch = Batch()
        temp_batch = temp_batch.from_data_list(raw_mask_sample)
        mask_sample = Data(x=temp_batch.x, edge_index=temp_batch.edge_index, edge_weight=temp_batch.edge_weight)
        self.mask.append(mask_sample)


def _build_prediction_mask(self, raw_mask):
    self.past_mask, self.future_mask = [], []
    for i in range(len(self)):
        raw_mask_sample = [Data(x=raw_mask[i][t], edge_index=self.edge_index, edge_weight=self.edge_weight) for t in
                           range(self.seq_len)]
        temp_batch = Batch()
        temp_batch = temp_batch.from_data_list(raw_mask_sample)
        mask_sample = Data(x=temp_batch.x, edge_index=temp_batch.edge_index, edge_weight=temp_batch.edge_weight)
        self.past_mask.append(mask_sample)
        raw_mask_sample = [Data(x=raw_mask[i][t], edge_index=self.edge_index, edge_weight=self.edge_weight) for t in
                           range(self.seq_len, self.seq_len * 2)]
        temp_batch = temp_batch.from_data_list(raw_mask_sample)
        mask_sample = Data(x=temp_batch.x, edge_index=temp_batch.edge_index, edge_weight=temp_batch.edge_weight)
        self.future_mask.append(mask_sample)


def _get_imputation_item(self, index):
    data_forward_sample = self.data[index]
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


def _get_prediction_item(self, index):
    past_data_sample = self.past_data[index]
    future_data_sample = self.future_data[index]
    past_mask_sample = self.past_mask[index]
    future_mask_sample = self.future_mask[index]
    return past_data_sample, future_data_sample, past_mask_sample, future_mask_sample


def _imputation_length(self):
    return max(self.num_timestamp - self.seq_len + 1, 0)


def _prediction_length(self):
    return max(self.num_timestamp - self.seq_len * 2 + 1, 0)


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
        tuple([self.generate_forwards_sample(self, self.data[i].numpy(), self.mask[i].numpy())
               for i in range(len(self))]), dim=0).float()
    self.forwards_b = torch.stack(
        tuple([self.generate_forwards_sample(self, self.data[i].numpy()[::-1],
                                             self.mask[i].numpy()[::-1]) for i in range(len(self))]), dim=0).float()


class LosAngelesHighwaySpeed(Dataset):
    def __init__(self, raw_data_dir, raw_adjacency_dir, missing_type, missing_rate, num_timestamp, train_proportion,
                 seq_len, graph_size, feature_channels):
        self.raw_data_dir = raw_data_dir
        self.raw_adjacency_dir = raw_adjacency_dir
        self.missing_type = missing_type
        self.missing_rate = missing_rate
        self.num_timestamp = num_timestamp
        self.train_proportion = train_proportion
        self.seq_len = seq_len
        self.graph_size = graph_size
        self.feature_channels = feature_channels
        self.element_num = self.graph_size * self.feature_channels
        self.random_walk_table = None
        if self.missing_type == 'random':
            self.generate_raw_mask_sample = _generate_random_raw_mask_sample
        elif self.missing_type == 'spatial':
            self.generate_raw_mask_sample = _generate_spatial_raw_mask_sample
        elif self.missing_type == 'temporal':
            self.generate_raw_mask_sample = _generate_temporal_raw_mask_sample
        elif self.missing_type == 'spatial-temporal':
            self.generate_raw_mask_sample = _generate_spatial_temporal_raw_mask_sample
        else:
            raise NotImplementedError
        raw_adjacency = self.load_raw_adjacency()
        self.build_adjacency(raw_adjacency)
        raw_data = self.load_raw_data(self)
        raw_mask = self.generate_raw_mask(self)
        self.build_data(self, raw_data)
        self.build_mask(self, raw_mask)

        self.data = [d.x.view(self.seq_len, self.graph_size) for d in self.data]
        self.mask = [d.x.view(self.seq_len, self.graph_size) for d in self.mask]

        self.generate_deltas_sample = parse_delta_sample
        self.generate_deltas(self)

        self.generate_forwards_sample = gen_forwards_sample
        self.generate_forwards(self)

    def _random_walk(self, num_missing_node):
        if self.random_walk_table == None:
            self.random_walk_table = [[] for _ in range(self.graph_size)]
            for root_node in range(self.graph_size):
                selected_node = np.array([False for _ in range(self.graph_size)])
                current_node = root_node
                self.random_walk_table[root_node].append(current_node)
                selected_node[current_node] = True
                while True:
                    node_prob = self.gaussian_adjacency[current_node] * (1 - selected_node)
                    if node_prob.sum() > 0:
                        next_node = node_prob.argmax()
                        current_node = next_node
                        self.random_walk_table[root_node].append(current_node)
                        selected_node[current_node] = True
                    else:
                        break
        selected_node = np.array([False for _ in range(self.graph_size)])
        num_selected_node = 0
        while num_selected_node < num_missing_node:
            current_node = np.random.choice(np.arange(self.graph_size)[~selected_node])
            walk_length = min(len(self.random_walk_table[current_node]), num_missing_node - num_selected_node)
            selected_node[self.random_walk_table[current_node][:walk_length]] = True
            num_selected_node += walk_length
        return selected_node

    def load_raw_data(self):
        raise NotImplementedError

    def load_raw_adjacency(self):
        raw_adjacency = np.load(self.raw_adjacency_dir)
        return raw_adjacency

    def generate_raw_mask(self):
        raise NotImplementedError

    def build_data(self, raw_data):
        raise NotImplementedError

    def build_adjacency(self, raw_adjacency):
        std = raw_adjacency[~np.isinf(raw_adjacency)].flatten().std()
        self.gaussian_adjacency = np.exp(-np.square(raw_adjacency / std))
        start_node, end_node = [], []
        self.edge_weight = []
        for i in range(self.graph_size):
            for j in range(self.graph_size):
                if self.gaussian_adjacency[i][j] >= 0.1:
                    start_node.append(i)
                    end_node.append(j)
                    self.edge_weight.append(self.gaussian_adjacency[i][j])
        self.edge_index = torch.tensor([start_node, end_node], dtype=torch.int64)
        self.edge_weight = torch.tensor(self.edge_weight).float()

    def build_mask(self, raw_mask):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class LosAngelesHighwaySpeedImputationTrainSet(LosAngelesHighwaySpeed):
    def __init__(self, raw_data_dir, raw_adjacency_dir, missing_type, missing_rate, num_timestamp, train_proportion,
                 seq_len, graph_size, feature_channels):
        self.load_raw_data = _load_train_raw_data
        self.generate_raw_mask = _generate_imputation_raw_mask
        self.build_data = _build_imputation_data
        self.build_mask = _build_imputation_mask
        self.generate_deltas = generate_imputation_deltas
        self.generate_forwards = generate_imputation_forwards
        super().__init__(raw_data_dir, raw_adjacency_dir, missing_type, missing_rate, num_timestamp, train_proportion,
                         seq_len, graph_size, feature_channels)

    __getitem__ = _get_imputation_item
    __len__ = _imputation_length


class LosAngelesHighwaySpeedImputationTestSet(LosAngelesHighwaySpeed):
    def __init__(self, raw_data_dir, raw_adjacency_dir, missing_type, missing_rate, num_timestamp, train_proportion,
                 seq_len, graph_size, feature_channels):
        self.load_raw_data = _load_test_raw_data
        self.generate_raw_mask = _generate_imputation_raw_mask
        self.build_data = _build_imputation_data
        self.build_mask = _build_imputation_mask
        self.generate_deltas = generate_imputation_deltas
        self.generate_forwards = generate_imputation_forwards
        super().__init__(raw_data_dir, raw_adjacency_dir, missing_type, missing_rate, num_timestamp, train_proportion,
                         seq_len, graph_size, feature_channels)

    __getitem__ = _get_imputation_item
    __len__ = _imputation_length


class LosAngelesHighwaySpeedPredictionTrainSet(LosAngelesHighwaySpeed):
    def __init__(self, raw_data_dir, raw_adjacency_dir, missing_type, missing_rate, num_timestamp, train_proportion,
                 seq_len, graph_size, feature_channels):
        self.load_raw_data = _load_train_raw_data
        self.generate_raw_mask = _generate_prediction_raw_mask
        self.build_data = _build_prediction_data
        self.build_mask = _build_prediction_mask
        super().__init__(raw_data_dir, raw_adjacency_dir, missing_type, missing_rate, num_timestamp, train_proportion,
                         seq_len, graph_size, feature_channels)

    __getitem__ = _get_prediction_item
    __len__ = _prediction_length


class LosAngelesHighwaySpeedPredictionTestSet(LosAngelesHighwaySpeed):
    def __init__(self, raw_data_dir, raw_adjacency_dir, missing_type, missing_rate, num_timestamp, train_proportion,
                 seq_len, graph_size, feature_channels):
        self.load_raw_data = _load_test_raw_data
        self.generate_raw_mask = _generate_prediction_raw_mask
        self.build_data = _build_prediction_data
        self.build_mask = _build_prediction_mask
        super().__init__(raw_data_dir, raw_adjacency_dir, missing_type, missing_rate, num_timestamp, train_proportion,
                         seq_len, graph_size, feature_channels)

    __getitem__ = _get_prediction_item
    __len__ = _prediction_length
