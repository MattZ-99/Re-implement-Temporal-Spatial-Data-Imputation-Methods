import torch

mask = torch.randint(0, 2, size=(64, 16, 32, 32))

a = torch.ones((64, 8, 32, 32))
b = torch.zeros((64, 8, 32, 32))
c = torch.cat((a, b), dim=1)
print(mask * c)

