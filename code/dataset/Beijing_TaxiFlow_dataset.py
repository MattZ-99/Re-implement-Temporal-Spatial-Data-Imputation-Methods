# -*- coding=utf-8 -*-


import numpy as np
import torch
from torch.utils.data import Dataset


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
	self.mask = torch.stack(tuple([torch.cat((self.generate_mask_sample(self), self.generate_mask_sample(self)), dim=0) for _ in range(len(self))]), dim=0)


def get_imputation_item(self, index):
	data_sample = self.data[index:index + self.seq_len, :, :, :]
	mask_sample = self.mask[index]
	return data_sample, mask_sample


def get_prediction_item(self, index):
	data_sample = self.data[index:index + self.seq_len * 2, :, :, :]
	mask_sample = self.mask[index]
	return data_sample, mask_sample


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
	spatial_mask[:, start_index_x:start_index_x + missing_length_x, start_index_y:start_index_y + missing_length_y, :].fill_(0)
	return spatial_mask


def generate_spatial_temporal_mask(self):
	spatial_temporal_mask = torch.ones(self.seq_len, self.grid_size_x, self.grid_size_y, self.feature_channels)
	missing_length_t = int(self.seq_len * (self.missing_rate ** (1 / 3)))
	missing_length_x = int(self.grid_size_x * (self.missing_rate ** (1 / 3)))
	missing_length_y = int(self.grid_size_y * (self.missing_rate ** (1 / 3)))
	start_index_t = np.random.randint(0, self.seq_len - missing_length_t + 1)
	start_index_x = np.random.randint(0, self.grid_size_x - missing_length_x + 1)
	start_index_y = np.random.randint(0, self.grid_size_y - missing_length_y + 1)
	spatial_temporal_mask[start_index_t:start_index_t + missing_length_t, start_index_x:start_index_x + missing_length_x, start_index_y:start_index_y + missing_length_y, :].fill_(0)
	return spatial_temporal_mask


class BeijingTaxiFlow(Dataset):
	def __init__(self, raw_data_dir, missing_type, missing_rate, num_timestamp, train_proportion, seq_len, grid_size_x, grid_size_y, feature_channels):
		self.raw_data_dir = raw_data_dir
		self.missing_type = missing_type
		self.missing_rate = missing_rate
		self.num_timestamp = num_timestamp
		self.train_proportion = train_proportion
		self.seq_len = seq_len
		self.grid_size_x = grid_size_x
		self.grid_size_y = grid_size_y
		self.feature_channels = feature_channels
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
	def load_data(self):
		raise NotImplementedError
	def generate_mask(self):
		raise NotImplementedError
	def __getitem__(self, index):
		raise NotImplementedError
	def __len__(self):
		raise NotImplementedError


class BeijingTaxiFlowImputationTrainSet(BeijingTaxiFlow):
	def __init__(self, raw_data_dir, missing_type, missing_rate, num_timestamp, train_proportion, seq_len, grid_size_x, grid_size_y, feature_channels):
		self.load_data = load_train_data
		self.generate_mask = generate_imputation_mask
		super().__init__(raw_data_dir, missing_type, missing_rate, num_timestamp, train_proportion, seq_len, grid_size_x, grid_size_y, feature_channels)
	__getitem__ = get_imputation_item
	__len__ = get_imputation_length


class BeijingTaxiFlowImputationTestSet(BeijingTaxiFlow):
	def __init__(self, raw_data_dir, missing_type, missing_rate, num_timestamp, train_proportion, seq_len, grid_size_x, grid_size_y, feature_channels):
		self.load_data = load_test_data
		self.generate_mask = generate_imputation_mask
		super().__init__(raw_data_dir, missing_type, missing_rate, num_timestamp, train_proportion, seq_len, grid_size_x, grid_size_y, feature_channels)
	__getitem__ = get_imputation_item
	__len__ = get_imputation_length


class BeijingTaxiFlowPredictionTrainSet(BeijingTaxiFlow):
	def __init__(self, raw_data_dir, missing_type, missing_rate, num_timestamp, train_proportion, seq_len, grid_size_x, grid_size_y, feature_channels):
		self.load_data = load_train_data
		self.generate_mask = generate_prediction_mask
		super().__init__(raw_data_dir, missing_type, missing_rate, num_timestamp, train_proportion, seq_len, grid_size_x, grid_size_y, feature_channels)
	__getitem__ = get_prediction_item
	__len__ = get_prediction_length


class BeijingTaxiFlowPredictionTestSet(BeijingTaxiFlow):
	def __init__(self, raw_data_dir, missing_type, missing_rate, num_timestamp, train_proportion, seq_len, grid_size_x, grid_size_y, feature_channels):
		self.load_data = load_test_data
		self.generate_mask = generate_prediction_mask
		super().__init__(raw_data_dir, missing_type, missing_rate, num_timestamp, train_proportion, seq_len, grid_size_x, grid_size_y, feature_channels)
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
	def __init__(self, raw_data_dir, missing_type, missing_rate, num_timestamp, train_proportion, seq_len, grid_size_x, grid_size_y, feature_channels):
		self.load_data = load_synthetic_train_data
		self.generate_mask = load_synthetic_train_mask
		super().__init__(raw_data_dir, missing_type, missing_rate, num_timestamp, train_proportion, seq_len, grid_size_x, grid_size_y, feature_channels)
	def __getitem__(self, index):
		return self.data[index], self.mask[index]
	def __len__(self):
		return self.data.size(0)


class BeijingTaxiFlowSyntheticTestDataset(BeijingTaxiFlow):
	def __init__(self, raw_data_dir, missing_type, missing_rate, num_timestamp, train_proportion, seq_len, grid_size_x, grid_size_y, feature_channels):
		self.load_data = load_synthetic_test_data
		self.generate_mask = load_synthetic_test_mask
		super().__init__(raw_data_dir, missing_type, missing_rate, num_timestamp, train_proportion, seq_len, grid_size_x, grid_size_y, feature_channels)
	def __getitem__(self, index):
		return self.data[index], self.mask[index]
	def __len__(self):
		return self.data.size(0)
