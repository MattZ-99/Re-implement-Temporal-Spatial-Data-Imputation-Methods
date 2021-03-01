# import models
#
# model = getattr(models, 'rits_i')
# print(model)

# direct = "backward"
#
# direct_flag = 1 if direct == "forward" else 0
#
# print(direct_flag)

import torch

a = torch.ones(64, 16, 1024)

b = a[:, 8:, :]

print(b.shape)