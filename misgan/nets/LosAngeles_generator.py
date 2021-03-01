import torch
import torch.nn as nn
import torch.nn.functional as F


def add_data_transformer(self):
    self.transform = lambda x: torch.sigmoid(x)


def add_mask_transformer(self, temperature=.66, hard_sigmoid=(-.1, 1.1)):
    """
    hard_sigmoid:
        False:  use sigmoid only
        True:   hard thresholding
        (a, b): hard thresholding on rescaled sigmoid
    """
    self.temperature = temperature
    self.hard_sigmoid = hard_sigmoid

    if hard_sigmoid is False:
        self.transform = lambda x: torch.sigmoid(x / temperature)
    elif hard_sigmoid is True:
        self.transform = lambda x: F.hardtanh(
            x / temperature, 0, 1)
    else:
        a, b = hard_sigmoid
        self.transform = lambda x: F.hardtanh(
            torch.sigmoid(x / temperature) * (b - a) + a, 0, 1)


class ConvGenerator(nn.Module):
    def __init__(self, latent_size=128):
        super().__init__()

        self.DIM = 64
        self.latent_size = latent_size

        self.preprocess = nn.Sequential(
            nn.Linear(latent_size, 4 * 5 * 5 * self.DIM),
            nn.ReLU(True),
        )
        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(4 * self.DIM, 2 * self.DIM, 3, padding=1),
            nn.ReLU(True),
        )
        self.block2 = nn.Sequential(
            nn.ConvTranspose2d(2 * self.DIM, self.DIM, 3),
            nn.ReLU(True),
        )
        self.deconv_out = nn.ConvTranspose2d(self.DIM, 1, 4, stride=2)

        self.fc = nn.Linear(256, 207)

    def forward(self, input):
        net = self.preprocess(input)
        net = net.view(-1, 4 * self.DIM, 5, 5)
        net = self.block1(net)
        net = net[:, :, :9, :9]
        net = self.block2(net)
        net = self.deconv_out(net)
        net = net.view(-1, 1, 256)
        net = self.fc(net)

        return self.transform(net)


class ConvDataGenerator(ConvGenerator):
    def __init__(self, latent_size=128):
        super().__init__(latent_size=latent_size)
        add_data_transformer(self)


class ConvMaskGenerator(ConvGenerator):
    def __init__(self, latent_size=128, temperature=.66,
                 hard_sigmoid=(-.1, 1.1)):
        super().__init__(latent_size=latent_size)
        add_mask_transformer(self, temperature, hard_sigmoid)
