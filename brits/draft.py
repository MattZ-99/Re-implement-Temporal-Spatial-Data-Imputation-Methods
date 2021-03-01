# # from torch.utils.data import DataLoader
# # from datasets.Beijing_TaxiFlow_dataset import BeijingTaxiFlowImputationTrainSet, BeijingTaxiFlowImputationTestSet
# # import numpy as np
# # import models
# #
# # model = getattr(models, 'rits_i')
# # print(model.__class__)
# # dataset = BeijingTaxiFlowImputationTestSet(raw_data_dir="./datasets/Beijing_TaxiFlow_data.npy",
# #                                             missing_type="random", missing_rate=0.25,
# #                                             num_timestamp=22272, train_proportion=0.95,
# #                                             seq_len=8, grid_size_x=32, grid_size_y=32, feature_channels=1)
# # #
# # # batch = dataset[0]
# # # print(type(batch), len(batch))
# # # # print(data_f.dtype, data_b.dtype, mask_f.dtype, mask_b.dtype, delta_f.dtype, delta_b.dtype, forwards_f.dtype, forwards_b.dtype)
# # # #
# # # # data_iter = DataLoader(dataset=dataset,
# # # #                        batch_size=8,
# # #                        num_workers=4,
# # #                        shuffle=True,
# # #                        pin_memory=True,
# # #                        )
# #
# # # for iter, batch in enumerate(data_iter):
# # #     data, mask, delta_f, delta_b, forwards_f, forwards_b = batch
# # #     print(data.shape)
# #
# #
# from datasets.LosAngeles_HighwaySpeed_dataset import LosAngelesHighwaySpeedImputationTrainSet, \
#     LosAngelesHighwaySpeedImputationTestSet
# #
# # train_set = LosAngelesHighwaySpeedImputationTrainSet(raw_data_dir="./datasets/LosAngeles_HighwaySpeed_data.npy",
# #                                                      raw_adjacency_dir="./datasets/LosAngeles_HighwaySpeed_adjacency"
# #                                                                        ".npy",
# #                                                      missing_type="random", missing_rate=0.25,
# #                                                      num_timestamp=34272, train_proportion=0.95,
# #                                                      seq_len=8, graph_size=207, feature_channels=1)
# #
# # valid_set = LosAngelesHighwaySpeedImputationTestSet(raw_data_dir="./datasets/LosAngeles_HighwaySpeed_data.npy",
# #                                                     raw_adjacency_dir="./datasets/LosAngeles_HighwaySpeed_adjacency"
# #                                                                       ".npy",
# #                                                     missing_type="random", missing_rate=0.25,
# #                                                     num_timestamp=34272, train_proportion=0.95,
# #                                                     seq_len=8, graph_size=207, feature_channels=1)
# #
# # train_data = train_set[0]
# #
# # data_forward_sample, data_backward_sample, mask_forward_sample, mask_backward_sample, \
# # deltas_forward_sample, deltas_backward_sample, forwards_forward_sample, forwards_backward_sample = train_data
# #
# # print(data_forward_sample.shape, data_backward_sample.shape)
#
# # valid_data = valid_set[0]
# #
# # print(len(valid_data), valid_data[5].shape)
#
#
# # import numpy as np
# # data = np.load("./datasets/Shenzhen_data.npy")
# # print(data.shape)
#
# # train_set = LosAngelesHighwaySpeedImputationTrainSet(raw_data_dir="./datasets/Shenzhen_data.npy",
# #                                                      raw_adjacency_dir="./datasets/Shenzhen_adjacency.npy",
# #                                                      missing_type="random", missing_rate=0.25,
# #                                                      num_timestamp=2970, train_proportion=0.95,
# #                                                      seq_len=8, graph_size=165, feature_channels=1)
# #
# # train_data = train_set[0]
# #
# # data_forward_sample, data_backward_sample, mask_forward_sample, mask_backward_sample, \
# # deltas_forward_sample, deltas_backward_sample, forwards_forward_sample, forwards_backward_sample = train_data
# #
# # print(data_forward_sample.shape, data_backward_sample.shape)

# from datasets.Beijing_TaxiFlow_dataset import BeijingTaxiFlowPredictionTrainSet
#
# train_set = BeijingTaxiFlowPredictionTrainSet(raw_data_dir="./datasets/Beijing_TaxiFlow_data.npy",
#                                               missing_type="random", missing_rate=0.25,
#                                               num_timestamp=22272, train_proportion=0.95,
#                                               seq_len=8, grid_size_x=32, grid_size_y=32, feature_channels=1)
# data, mask = train_set[0]
# print(data.shape, mask.shape)

import numpy as np

a = np.array([1, 2, 3, 4, 5])
b = np.array([1, 1, 0, 0, 1])
c = np.where(b, a, np.nan)
print(c)
