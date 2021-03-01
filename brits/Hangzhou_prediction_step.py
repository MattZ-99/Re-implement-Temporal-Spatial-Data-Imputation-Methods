import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import data_preprocess
from datasets.LosAngeles_HighwaySpeed_dataset_prediction import LosAngelesHighwaySpeedPredictionTrainSet, \
    LosAngelesHighwaySpeedPredictionTestSet
from tools.Parser import get_phaser
from tools import utils, metric
import models_predict as models
import numpy as np
from tqdm import tqdm
import os

args = get_phaser().parse_args()
print("args:", args)

utils.setup_seed(args.seed)

# model = getattr(models, args.model).Model(args.hid_size, args.seq_len, args.element_num)
# total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print('Total params is {}'.format(total_params))
# print(type(model))
# checkpoint = torch.load("./outputs/Predict_Hangzhou_data_v2/brits/20210204_194136_mr_0.25_mt_spatial-temporal/models/epoch_290_lr_0.001.pth")
# model.load_state_dict(checkpoint["brits"])

args.cuda_is_available = torch.cuda.is_available()
model = torch.load("./outputs/Predict_Hangzhou_data_v2/brits/20210204_213105_mr_0.75_mt_spatial-temporal/models/epoch_290_lr_0.001.pth")["brits"]

if args.cuda_is_available:
    model = model.cuda()
output_path = "./outputs/predict_step/"
utils.makedirs(output_path)


print("==============>Valid set loading")
valid_set = LosAngelesHighwaySpeedPredictionTestSet(raw_data_dir="./datasets/Hangzhou_data.npy",
                                                    raw_adjacency_dir="./datasets/Hangzhou_adjacency.npy",
                                                    missing_type=args.missing_type, missing_rate=args.missing_rate,
                                                    num_timestamp=2625, train_proportion=0.95,
                                                    seq_len=8, graph_size=81, feature_channels=1)

valid_dataloader = DataLoader(dataset=valid_set, batch_size=args.batch_size, num_workers=args.num_workers,
                              shuffle=True, drop_last=True)

print("==============>Valid mask loading")
valid_mask_set = LosAngelesHighwaySpeedPredictionTestSet(raw_data_dir="./datasets/Hangzhou_data.npy",
                                                         raw_adjacency_dir="./datasets/Hangzhou_adjacency.npy",
                                                         missing_type=args.missing_type, missing_rate=args.missing_rate,
                                                         num_timestamp=2625, train_proportion=0.95,
                                                         seq_len=8, graph_size=81, feature_channels=1)

valid_mask_dataloader = DataLoader(dataset=valid_mask_set, batch_size=args.batch_size, num_workers=args.num_workers,
                                   shuffle=True, drop_last=True)


model.eval()

mae = np.zeros(8)
rmse = np.zeros(8)
num = 0
with torch.no_grad():
    mask_zero = torch.zeros(args.batch_size, 1, 81)
    if args.cuda_is_available:
        mask_zero = mask_zero.cuda()
    valid_bar = tqdm(zip(valid_dataloader, valid_mask_dataloader), total=int(len(valid_set) / args.batch_size))
    for batch, batch_for_mask in valid_bar:
        mask_gen = batch_for_mask[2][:, args.seq_len:, :]
        if args.cuda_is_available:
            batch = [item.cuda() for item in batch]
            mask_gen = mask_gen.cuda()
        forward = {'values': batch[0], 'masks': batch[2], 'deltas': batch[4], 'forwards': batch[6]}
        backward = {'values': batch[1], 'masks': batch[3], 'deltas': batch[5], 'forwards': batch[7]}
        data = {'forward': forward, 'backward': backward}

        ret = model.run_on_batch(data, None)

        real_data = batch[0][:, args.seq_len:, :]
        mask = batch[2][:, args.seq_len:, :]
        imputed_data = ret['imputations'].data[:, args.seq_len:, :]

        real_data = data_preprocess.scaler_for_impute(valid_set.scaler, real_data)
        imputed_data = data_preprocess.scaler_for_impute(valid_set.scaler, imputed_data)


        for i in range(args.seq_len):
            rd = real_data[:, i, :]
            id = imputed_data[:, i, :]
            mae1 = metric.compute_mean_absolute_error(rd, id, mask_zero)
            rmse1 = metric.compute_root_mean_square_error(rd, id, mask_zero)

            mae[i] += mae1
            rmse[i] += rmse1

        num += 1
print(num, int(len(valid_set) / args.batch_size))
print(mae/num)
print(rmse/num)