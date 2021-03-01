import torch


def predict_mask_zero(mask):
    N, C, H, W = mask.shape
    seq_len = int(C/2)
    a = torch.ones((N, seq_len, H, W))
    b = torch.zeros((N, C - seq_len, H, W))
    c = torch.cat((a, b), dim=1)

    if torch.cuda.is_available():
        c = c.cuda()

    return mask * c