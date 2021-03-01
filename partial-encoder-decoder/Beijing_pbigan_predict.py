import torch
from torch.utils.data import DataLoader
from torch import optim

from tools.Parse import get_phaser
from tools import utils, metric

from nets_predict.Beijing_decoder import ConvDecoder
from nets_predict.Beijing_encoder import ConvEncoder
from nets_predict.PBIGAN import PBiGAN
from nets_predict.Beijing_critic import ConvCritic
from nets_predict.GradientPenalty import GradientPenalty
from dataset import data_preprocess
from dataset.Beijing_TaxiFlow_dataset import BeijingTaxiFlowPredictionTrainSet, BeijingTaxiFlowPredictionTestSet

from tqdm import tqdm
import os

# set gpu env
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
# set all random seed
utils.setup_seed(210122)

# get input parser
args = get_phaser().parse_args()
print("args:", args)

# path
name = "Predict_Beijing_misgan_{}_mr_{}_mt_{}".format(
    utils.time_for_save(), args.missing_rate, args.missing_type)
model_path = "./outputs_predict/{}/models/".format(name)
plot_path = "./outputs_predict/{}/visualization/".format(name)
log_path = "./outputs_predict/{}/logs/".format(name)
utils.makedirs(log_path)
utils.makedirs(model_path)
utils.makedirs(plot_path)

# dataset loader
print("=============>Initializing dataset")

trainset_Beijing = BeijingTaxiFlowPredictionTrainSet(raw_data_dir="./dataset/Beijing_TaxiFlow_data.npy",
                                                     missing_type=args.missing_type, missing_rate=args.missing_rate,
                                                     num_timestamp=22272, train_proportion=0.95,
                                                     seq_len=8, grid_size_x=32, grid_size_y=32, feature_channels=1)
train_data_loader = DataLoader(trainset_Beijing, batch_size=args.batch_size, num_workers=args.num_workers,
                               shuffle=True, drop_last=True)
train_mask_loader = DataLoader(trainset_Beijing, batch_size=args.batch_size, num_workers=args.num_workers,
                               shuffle=True, drop_last=True)

validset_Beijing = BeijingTaxiFlowPredictionTestSet(raw_data_dir="./dataset/Beijing_TaxiFlow_data.npy",
                                                    missing_type=args.missing_type, missing_rate=args.missing_rate,
                                                    num_timestamp=22272, train_proportion=0.95,
                                                    seq_len=8, grid_size_x=32, grid_size_y=32, feature_channels=1)

valid_data_loader = DataLoader(validset_Beijing, batch_size=args.batch_size, num_workers=args.num_workers,
                               shuffle=True, drop_last=True)
valid_mask_loader = DataLoader(validset_Beijing, batch_size=args.batch_size, num_workers=args.num_workers,
                               shuffle=True, drop_last=True)

print("=============>Initializing network")
decoder = ConvDecoder(args.latent, args.time_stamp_length * 2)
encoder = ConvEncoder(args.latent, args.flow, logprob=False, input_channel=args.time_stamp_length * 2)
pbigan = PBiGAN(encoder, decoder, args.aeloss).to(device)

critic = ConvCritic(args.latent, args.time_stamp_length * 2).to(device)
grad_penalty = GradientPenalty(critic, args.batch_size, device=device)

# optimizer
print("=============>Initializing optimizer")
lrate = 1e-4
optimizer = optim.Adam(pbigan.parameters(), lr=lrate, betas=(.5, .9))

critic_optimizer = optim.Adam(
    critic.parameters(), lr=lrate, betas=(.5, .9))

# lr scheduler
scheduler = utils.make_scheduler(optimizer, args.lr, args.min_lr, args.epoch)

n_critic = 5
critic_updates = 0
ae_weight = 0
ae_flat = 100

# some output value
D_loss_value = utils.ValueStat()
G_AE_loss_value = utils.ValueStat()
metric_mae = utils.ValueStat()
metric_rmse = utils.ValueStat()

D_loss_curve = utils.ValuesVisual()
G_AE_loss_curve = utils.ValuesVisual()
mae_curve = utils.ValuesVisual()
rmse_curve = utils.ValuesVisual()
distance_curve = utils.ValuesVisual()
partial_WD_curve = utils.ValuesVisual()

print("=============>Training begin")
for epoch in range(args.epoch):
    if epoch > ae_flat:
        ae_weight = args.ae * (epoch - ae_flat) / (args.epoch - ae_flat)

    # train
    pbigan.train()
    # reset values
    D_loss_value.reset()
    G_AE_loss_value.reset()
    bar_total = len(train_data_loader)
    train_bar = tqdm(zip(train_data_loader, train_mask_loader), desc="[Train] eopch={}".format(epoch),
                     total=bar_total)

    for (data, mask), (_, mask_gen) in train_bar:
        data = data.to(device)
        mask = mask.to(device).float()
        mask_gen = mask_gen.to(device).float()

        z_enc, z_gen, x_rec, x_gen, _ = pbigan(data, mask, ae=False)

        real_score = critic((data * mask, z_enc)).mean()
        fake_score = critic((x_gen * mask_gen, z_gen)).mean()

        w_dist = real_score - fake_score
        D_loss = -w_dist + grad_penalty((data * mask, z_enc),
                                        (x_gen * mask_gen, z_gen))

        critic_optimizer.zero_grad()
        D_loss.backward()
        critic_optimizer.step()
        D_loss_value.update(D_loss.data)

        critic_updates += 1
        if critic_updates == n_critic:
            critic_updates = 0

            for p in critic.parameters():
                p.requires_grad_(False)

            z_enc, z_gen, x_rec, x_gen, ae_loss = pbigan(data, mask)

            real_score = critic((data * mask, z_enc)).mean()
            fake_score = critic((x_gen * mask_gen, z_gen)).mean()

            G_loss = real_score - fake_score

            ae_loss = ae_loss * ae_weight
            loss = G_loss + ae_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            G_AE_loss_value.update(loss.data)

            for p in critic.parameters():
                p.requires_grad_(True)

    if scheduler:
        scheduler.step()

    output = "[Train] epoch={}, D_loss={}, G_AE_loss={}".format(epoch, D_loss_value.get_avg(), G_AE_loss_value.get_avg())
    print(output)
    utils.save_log(log_path + 'train' + '.log', output + '\n')

    D_loss_curve.add_value(D_loss_value.get_avg())
    G_AE_loss_curve.add_value(G_AE_loss_value.get_avg())
    torch.cuda.empty_cache()

    # eval
    pbigan.eval()
    metric_mae.reset()
    metric_rmse.reset()
    list_real_data = []
    list_predict_data = []
    list_masked_real_data = []
    list_masked_predict_data = []
    with torch.no_grad():
        mask_zeros = torch.zeros(args.batch_size, args.time_stamp_length, 32, 32)
        if torch.cuda.is_available():
            mask_zeros = mask_zeros.cuda()
        bar_total = len(valid_data_loader)
        valid_bar = tqdm(zip(valid_data_loader, valid_mask_loader), desc="[Valid] eopch={}".format(epoch),
                         total=bar_total)
        for (data, mask), (_, mask_gen) in valid_bar:
            data = data.to(device)
            mask = mask.to(device).float()
            mask_gen = mask_gen.to(device).float()
            _, _, data_rec, _, _ = pbigan(data, mask, ae=False)

            data_predict = data_rec[:, args.time_stamp_length:, :, :]
            data = data[:, args.time_stamp_length:, :, :]
            mask = mask[:, args.time_stamp_length:, :, :]
            mask_gen = mask_gen[:, args.time_stamp_length:, :, :]

            real_data = data_preprocess.scaler_for_impute(validset_Beijing.scaler, data)
            predict_data = data_preprocess.scaler_for_impute(validset_Beijing.scaler, data_predict)

            masked_real_data = utils.mask_data(real_data, mask, 0)
            masked_predict_data = utils.mask_data(predict_data, mask_gen, 0)

            mae = metric.compute_mean_absolute_error(real_data, predict_data, mask_zeros)
            rmse = metric.compute_root_mean_square_error(real_data, predict_data, mask_zeros)
            metric_mae.update(mae)
            metric_rmse.update(rmse)

            list_masked_real_data.append(masked_real_data)
            list_masked_predict_data.append(masked_predict_data)

            list_real_data.append(real_data)
            list_predict_data.append(predict_data)

        masked_real_data = torch.cat(list_masked_real_data)
        masked_predict_data = torch.cat(list_masked_predict_data)
        all_real_data = torch.cat(list_real_data)
        all_predict_data = torch.cat(list_predict_data)
        first_wasserstein_distance = metric.compute_first_wasserstein_distance(all_real_data, all_predict_data)
        partial_WD = metric.compute_first_wasserstein_distance(masked_real_data, masked_predict_data)
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


    # model save and figure plot
    if epoch % 10 == 0:
        state = {'epoch': epoch,
                 'pbigan': pbigan.state_dict(),
                 'critic': critic.state_dict(),
                 }
        torch.save(state, os.path.join(model_path, "epoch_{}_lr_{}.pth".format(
            epoch, optimizer.param_groups[0]['lr']
                )
            )
        )

        D_loss_curve.plot(plot_path + 'D_loss' + '.jpg')
        G_AE_loss_curve.plot(plot_path + 'G_AE_loss' + '.jpg')
        mae_curve.plot(plot_path + 'mae' + '.jpg')
        rmse_curve.plot(plot_path + 'rmse' + '.jpg')
        distance_curve.plot(plot_path + 'complete_distance' + '.jpg')
        partial_WD_curve.plot(plot_path + 'partial_WD_distance' + '.jpg')
