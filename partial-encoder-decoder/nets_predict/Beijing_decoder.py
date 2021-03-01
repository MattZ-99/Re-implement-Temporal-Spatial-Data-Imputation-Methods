import torch
import torch.nn as nn


class ConvDecoder(nn.Module):
    def __init__(self, latent_size=128, output_channel=1):
        super().__init__()

        self.output_channel = output_channel
        self.DIM = 64
        self.latent_size = latent_size

        self.preprocess = nn.Sequential(
            nn.Linear(latent_size, 4 * 4 * 4 * self.DIM),
            nn.ReLU(True),
        )
        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(4 * self.DIM, 2 * self.DIM, 5),
            nn.ReLU(True),
        )
        self.block2 = nn.Sequential(
            nn.ConvTranspose2d(2 * self.DIM, self.DIM, 6),
            nn.ReLU(True),
        )
        self.deconv_out = nn.ConvTranspose2d(self.DIM, self.output_channel, 8, stride=2)

    def forward(self, input):
        net = self.preprocess(input)
        net = net.view(-1, 4 * self.DIM, 4, 4)
        net = self.block1(net)
        net = net[:, :, :8, :8]
        net = self.block2(net)
        net = self.deconv_out(net)
        net = net.view(-1, self.output_channel, 32, 32)
        return net, torch.sigmoid(net)
