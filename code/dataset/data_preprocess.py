import numpy as np
import torch

def scaler_for_impute(scaler, impute_data):
    impute_data = impute_data.cpu().numpy()
    N, C, H, W = impute_data.shape
    impute_data = impute_data.reshape(N, C*H*W)
    impute_data = scaler.inverse_transform(impute_data)
    impute_data = impute_data.reshape(N, C, H, W)
    impute_data = torch.from_numpy(impute_data).cuda()
    return impute_data