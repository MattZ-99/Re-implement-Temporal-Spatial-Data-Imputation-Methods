import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import transforms

from dataset import data_preprocess
from dataset.Beijing_Taxi_dataset import Beijing_taxi_impute_dataset_without_timestamp
from nets.Beijing_critic import ConvCritic, FCCritic
from nets.Beijing_generator import (ConvDataGenerator, FCDataGenerator, ConvMaskGenerator, FCMaskGenerator)
from nets.Beijing_imputer import ComplementImputer
from tools import utils, metric
from tools.utils import CriticUpdater

from tqdm import tqdm
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

utils.setup_seed(210118)

print("=============>Parameter loading")
# Discriminator
Critic = ConvCritic

data_critic = Critic().cuda()
mask_critic = Critic().cuda()
impu_critic = Critic().cuda()

# Generator
DataGenerator = ConvDataGenerator
MaskGenerator = ConvMaskGenerator

data_gen = DataGenerator().cuda()
mask_gen = MaskGenerator(hard_sigmoid=False).cuda()

# imputer
Imputer = ComplementImputer

imputer = Imputer(arch=[1024, 1024]).cuda()


pretrain_path = "./outputs/Beijing_misgan_20210120_220332_mr_0.5_mt_random_geninter_5/models/epoch_490_lr_0.0001.pth"
pretrain = torch.load(pretrain_path, map_location='cpu')
data_gen.load_state_dict(pretrain['data_gen'])
mask_gen.load_state_dict(pretrain['mask_gen'])
data_critic.load_state_dict(pretrain['data_critic'])
mask_critic.load_state_dict(pretrain['mask_critic'])

# parameter
batch_size = 16
nz = 128
gp_lambda = 1
tau = 0
epochs = 500
n_critic = 5

lrate = 1e-4
imputer_lrate = 2e-4
missing_rate = 0.5
missing_type = "random"

# path
name = "Beijing_misgan_imputeonly_{}_mr_{}_mt_{}_impuinter_{}".format(
    utils.time_for_save(), missing_rate, missing_type, n_critic)
model_path = "./outputs/{}/models/".format(name)
plot_path = "./outputs/{}/visualization/".format(name)
log_path = "./outputs/{}/logs/".format(name)
utils.makedirs(log_path)
utils.makedirs(model_path)
utils.makedirs(plot_path)


train_set = Beijing_taxi_impute_dataset_without_timestamp(
    transforms=transforms.ToTensor(), missing_rate=missing_rate, train_proportion=0.95,
    missing_type=missing_type
    )
train_data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)

valid_set = Beijing_taxi_impute_dataset_without_timestamp(mode="valid",
    transforms=transforms.ToTensor(), missing_rate=missing_rate, train_proportion=0.95,
    missing_type=missing_type
    )
valid_data_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, drop_last=True)


data_shape = train_set[0][0].shape
data_noise = torch.FloatTensor(batch_size, nz).cuda()
mask_noise = torch.FloatTensor(batch_size, nz).cuda()
impu_noise = torch.FloatTensor(batch_size, *data_shape).cuda()

eps = torch.FloatTensor(batch_size, 1, 1, 1).cuda()
ones = torch.ones(batch_size).cuda()

data_gen_optimizer = optim.Adam(
    data_gen.parameters(), lr=lrate, betas=(.5, .9))
mask_gen_optimizer = optim.Adam(
    mask_gen.parameters(), lr=lrate, betas=(.5, .9))
imputer_optimizer = optim.Adam(
    imputer.parameters(), lr=imputer_lrate, betas=(.5, .9))

data_critic_optimizer = optim.Adam(
    data_critic.parameters(), lr=lrate, betas=(.5, .9))
mask_critic_optimizer = optim.Adam(
    mask_critic.parameters(), lr=lrate, betas=(.5, .9))
impu_critic_optimizer = optim.Adam(
    impu_critic.parameters(), lr=imputer_lrate, betas=(.5, .9))

update_data_critic = CriticUpdater(
    data_critic, data_critic_optimizer, eps, ones, gp_lambda)
update_mask_critic = CriticUpdater(
    mask_critic, mask_critic_optimizer, eps, ones, gp_lambda)
update_impu_critic = CriticUpdater(
    impu_critic, impu_critic_optimizer, eps, ones, gp_lambda)


print("=============>Training")

start_epoch = 0
critic_updates = 0

impute_critic_loss_curve = utils.ValuesVisual()
imputer_loss_curve = utils.ValuesVisual()
mae_curve = utils.ValuesVisual()
rmse_curve = utils.ValuesVisual()
distance_curve = utils.ValuesVisual()
partial_distance_curve = utils.ValuesVisual()

for epoch in range(start_epoch, epochs):

    # train
    data_gen.train()
    imputer.train()
    impu_critic.train()

    impute_critic_loss = utils.ValueStat()
    imputer_loss = utils.ValueStat()
    for real_data, real_mask in tqdm(train_data_loader, desc="[Train] epoch={}".format(epoch)):
        real_data = real_data.float()
        real_mask = real_mask.float()
        real_data = real_data.cuda()
        real_mask = real_mask.cuda()

        masked_real_data = utils.mask_data(real_data, real_mask, tau)

        data_noise.normal_()
        fake_data = data_gen(data_noise)

        impu_noise.uniform_()
        imputed_data = imputer(real_data, real_mask, impu_noise)
        masked_imputed_data = utils.mask_data(real_data, real_mask, imputed_data)

        update_impu_critic(fake_data, masked_imputed_data)
        impute_critic_loss.update(update_impu_critic.loss_value)
        critic_updates += 1

        if critic_updates == n_critic:
            critic_updates = 0

            for p in impu_critic.parameters():
                p.requires_grad_(False)

            impu_noise.uniform_()
            imputed_data = imputer(real_data, real_mask, impu_noise)
            masked_imputed_data = utils.mask_data(real_data, real_mask, imputed_data)
            impu_loss = -impu_critic(masked_imputed_data).mean()
            imputer.zero_grad()
            impu_loss.backward()
            imputer_optimizer.step()

            imputer_loss.update(impu_loss)

            for p in impu_critic.parameters():
                p.requires_grad_(True)

    output = "[Train] epoch={}, impute_critic_loss={}, imputer_loss={}".format(
        epoch, impute_critic_loss.get_avg(), imputer_loss.get_avg())
    print(output)
    utils.save_log(log_path + 'train' + '.log', output + '\n')
    impute_critic_loss_curve.add_value(impute_critic_loss.get_avg())
    imputer_loss_curve.add_value(imputer_loss.get_avg())

    # eval
    imputer.eval()
    mask_gen.eval()
    metric_mae = utils.ValueStat()
    metric_rmse = utils.ValueStat()
    list_real_data = []
    list_impute_data = []
    list_masked_real_data = []
    list_masked_impute_data = []
    with torch.no_grad():
        valid_bar = tqdm(valid_data_loader, desc="[Valid] epoch={}".format(epoch))
        for real_data, real_mask in valid_bar:
            real_data = real_data.float()
            real_mask = real_mask.float()
            real_data = real_data.cuda()
            real_mask = real_mask.cuda()

            impu_noise.uniform_()
            imputed_data = imputer(real_data, real_mask, impu_noise)
            complete_imputed_data = utils.mask_data(real_data, real_mask, imputed_data)

            mask_noise.normal_()
            fake_mask = mask_gen(mask_noise)
            partial_imputed_data = utils.mask_data(complete_imputed_data, fake_mask, tau=tau)
            partial_real_data = utils.mask_data(real_data, real_mask, tau=tau)

            real_data = data_preprocess.scaler_for_impute(valid_set.scaler, real_data)
            complete_imputed_data = data_preprocess.scaler_for_impute(valid_set.scaler, complete_imputed_data)
            partial_real_data = data_preprocess.scaler_for_impute(valid_set.scaler, partial_real_data)
            partial_imputed_data = data_preprocess.scaler_for_impute(valid_set.scaler, partial_imputed_data)

            mae = metric.compute_mean_absolute_error(real_data, complete_imputed_data, real_mask)
            rmse = metric.compute_root_mean_square_error(real_data, complete_imputed_data, real_mask)
            list_real_data.append(real_data)
            list_impute_data.append(complete_imputed_data)
            list_masked_real_data.append(partial_real_data)
            list_masked_impute_data.append(partial_imputed_data)
            metric_mae.update(mae)
            metric_rmse.update(rmse)
            # valid_bar.set_postfix({"MAE": "{:.4f}".format(metric_mae.get_avg()),
            #                        "RMSE": "{:.4f}".format(metric_rmse.get_avg())
            #                        })
        all_real_data = torch.cat(list_real_data)
        all_impute_data = torch.cat(list_impute_data)
        all_partial_real_data = torch.cat(list_masked_real_data)
        all_partial_impute_data = torch.cat(list_masked_impute_data)
        first_wasserstein_distance = metric.compute_first_wasserstein_distance(all_real_data, all_impute_data)
        partial_WD = metric.compute_first_wasserstein_distance(all_partial_real_data, all_partial_impute_data)
        output = "[Valid] epoch={}, MAE={}, RMSE={}, complete distance={}, partial_WD={}".format(
            epoch, metric_mae.get_avg(), metric_rmse.get_avg(), first_wasserstein_distance, partial_WD
        )
        print(output)
        utils.save_log(log_path + 'valid' + '.log', output + '\n')
        mae_curve.add_value(metric_mae.get_avg())
        rmse_curve.add_value(metric_rmse.get_avg())
        distance_curve.add_value(first_wasserstein_distance)
        partial_distance_curve.add_value(partial_WD)
    if epoch % 10 == 0:
        state = {'epoch': epoch,
                 'data_critic': data_critic.state_dict(),
                 'mask_critic': mask_critic.state_dict(),
                 'impute_critic': impu_critic.state_dict(),
                 'data_gen': data_gen.state_dict(),
                 'mask_gen': mask_gen.state_dict(),
                 'imputer': imputer.state_dict()
                 }
        torch.save(state, os.path.join(model_path, "epoch_{}_lr_{}.pth".format(
            epoch, data_critic_optimizer.param_groups[0]['lr']
                )
            )
        )

        impute_critic_loss_curve.plot(plot_path + 'impute_critic_loss' + '.jpg')
        imputer_loss_curve.plot(plot_path + 'imputer_loss' + '.jpg')
        mae_curve.plot(plot_path + 'mae_curve' + '.jpg')
        rmse_curve.plot(plot_path + 'rmse_curve' + '.jpg')
        distance_curve.plot(plot_path + 'first_wasserstein_distance' + '.jpg')
        partial_distance_curve.plot(plot_path + 'partial_WD' + '.jpg')