import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import data_preprocess
from datasets.Beijing_TaxiFlow_dataset_predict import BeijingTaxiFlowPredictionTrainSet, BeijingTaxiFlowPredictionTestSet
from tools.Parser import get_phaser
from tools import utils, metric
import models_predict as models

from tqdm import tqdm
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


args = get_phaser().parse_args()
print("args:", args)

utils.setup_seed(args.seed)

model = getattr(models, args.model).Model(args.hid_size, args.seq_len, args.element_num)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Total params is {}'.format(total_params))

args.cuda_is_available = torch.cuda.is_available()
if args.cuda_is_available:
    model = model.cuda()

optimizer = optim.Adam(model.parameters(), lr=1e-3)

print("==============>Train set loading")
train_set = BeijingTaxiFlowPredictionTrainSet(raw_data_dir="./datasets/Beijing_TaxiFlow_data.npy",
                                             missing_type=args.missing_type, missing_rate=args.missing_rate,
                                             num_timestamp=22272, train_proportion=0.95,
                                             seq_len=args.seq_len, grid_size_x=32, grid_size_y=32, feature_channels=1)

train_dataloader = DataLoader(dataset=train_set, batch_size=args.batch_size, num_workers=args.num_workers,
                              shuffle=True, drop_last=True)

print("==============>Valid set loading")
valid_set = BeijingTaxiFlowPredictionTestSet(raw_data_dir="./datasets/Beijing_TaxiFlow_data.npy",
                                             missing_type=args.missing_type, missing_rate=args.missing_rate,
                                             num_timestamp=22272, train_proportion=0.95,
                                             seq_len=args.seq_len, grid_size_x=32, grid_size_y=32, feature_channels=1)

valid_dataloader = DataLoader(dataset=valid_set, batch_size=args.batch_size, num_workers=args.num_workers,
                              shuffle=True, drop_last=True)

print("==============>Valid mask loading")
valid_mask_set = BeijingTaxiFlowPredictionTestSet(raw_data_dir="./datasets/Beijing_TaxiFlow_data.npy",
                                                  missing_type=args.missing_type, missing_rate=args.missing_rate,
                                                  num_timestamp=22272, train_proportion=0.95,
                                                  seq_len=args.seq_len, grid_size_x=32, grid_size_y=32,
                                                  feature_channels=1)

valid_mask_dataloader = DataLoader(dataset=valid_mask_set, batch_size=args.batch_size, num_workers=args.num_workers,
                                   shuffle=True, drop_last=True)


name = "{}/{}_mr_{}_mt_{}".format(
    args.model, utils.time_for_save(), args.missing_rate, args.missing_type)
version = 1
dataset_name = "Predict_Beijing_data_v{}".format(version)
model_path = "./outputs/{}/{}/models/".format(dataset_name, name)
plot_path = "./outputs/{}/{}/visualization/".format(dataset_name, name)
log_path = "./outputs/{}/{}/logs/".format(dataset_name, name)
utils.makedirs(log_path)
utils.makedirs(model_path)
utils.makedirs(plot_path)


loss_curve = utils.ValuesVisual()
mae_curve = utils.ValuesVisual()
rmse_curve = utils.ValuesVisual()
distance_curve = utils.ValuesVisual()
partial_WD_curve = utils.ValuesVisual()
loss_value = utils.ValueStat()
metric_mae = utils.ValueStat()
metric_rmse = utils.ValueStat()

for epoch in range(args.epochs):
    model.train()
    loss_value.reset()
    for idx, batch in tqdm(enumerate(train_dataloader), total=int(len(train_set) / args.batch_size)):

        if args.cuda_is_available:
            batch = [item.cuda() for item in batch]
        forward = {'values': batch[0], 'masks': batch[2], 'deltas': batch[4], 'forwards': batch[6]}
        backward = {'values': batch[1], 'masks': batch[3], 'deltas': batch[5], 'forwards': batch[7]}
        data = {'forward': forward, 'backward': backward}
        ret = model.run_on_batch(data, optimizer, epoch)
        loss_value.update(ret['loss'].data)
    output = "[Train] epoch={}, loss={}".format(epoch, loss_value.get_avg())
    print(output)
    utils.save_log(log_path + 'train' + '.log', output + '\n')
    loss_curve.add_value(loss_value.get_avg())
    torch.cuda.empty_cache()

    model.eval()
    metric_mae.reset()
    metric_rmse.reset()
    list_real_data = []
    list_impute_data = []
    list_masked_real_data = []
    list_masked_impute_data = []
    with torch.no_grad():

        mask_zero = torch.zeros(args.batch_size, args.seq_len, 1024)
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
            imputed_data = ret['imputations'].data
            imputed_data = imputed_data[:, args.seq_len:, :]

            real_data = data_preprocess.scaler_for_impute(valid_set.scaler, real_data)
            imputed_data = data_preprocess.scaler_for_impute(valid_set.scaler, imputed_data)

            masked_real_data = utils.mask_data(real_data, mask, 0)
            masked_imputed_data = utils.mask_data(imputed_data, mask_gen, 0)

            list_masked_real_data.append(masked_real_data)
            list_masked_impute_data.append(masked_imputed_data)

            mae = metric.compute_mean_absolute_error(real_data, imputed_data, mask_zero)
            rmse = metric.compute_root_mean_square_error(real_data, imputed_data, mask_zero)
            metric_mae.update(mae)
            metric_rmse.update(rmse)

            list_real_data.append(real_data)
            list_impute_data.append(imputed_data)

    all_real_data = torch.cat(list_real_data)
    all_impute_data = torch.cat(list_impute_data)
    masked_real_data = torch.cat(list_masked_real_data)
    masked_impute_data = torch.cat(list_masked_impute_data)
    first_wasserstein_distance = metric.compute_first_wasserstein_distance(all_real_data, all_impute_data)
    partial_WD = metric.compute_first_wasserstein_distance(masked_real_data, masked_impute_data)
    output = "[Valid] epoch={}, MAE={}, RMSE={}, complete distance={}, partial_WD={}".format(
        epoch, metric_mae.get_avg(), metric_rmse.get_avg(), first_wasserstein_distance, partial_WD
    )
    print(output)
    utils.save_log(log_path + 'valid' + '.log', output + '\n')
    mae_curve.add_value(metric_mae.get_avg())
    rmse_curve.add_value(metric_rmse.get_avg())
    distance_curve.add_value(first_wasserstein_distance)
    partial_WD_curve.add_value(partial_WD)
    torch.cuda.empty_cache()

    if epoch % 10 == 0:
        state = {'epoch': epoch,
                 args.model: model
                 }
        torch.save(state, os.path.join(model_path, "epoch_{}_lr_{}.pth".format(
            epoch, optimizer.param_groups[0]['lr']
                )
            )
        )

        loss_curve.plot(plot_path + 'loss' + '.jpg')
        mae_curve.plot(plot_path + 'mae' + '.jpg')
        rmse_curve.plot(plot_path + 'rmse' + '.jpg')
        distance_curve.plot(plot_path + 'complete_distance' + '.jpg')
        partial_WD_curve.plot(plot_path + 'partial_WD_distance' + '.jpg')