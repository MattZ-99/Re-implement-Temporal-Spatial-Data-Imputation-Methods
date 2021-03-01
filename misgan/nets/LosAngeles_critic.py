import torch.nn as nn


class ConvCritic(nn.Module):
    def __init__(self):
        super().__init__()

        self.DIM = 64
        self.preprocess = nn.Sequential(
            nn.Linear(207, 256),
            nn.ReLU(True),
        )
        main = nn.Sequential(
            nn.Conv2d(1, self.DIM, 5, stride=2, padding=2),
            nn.ReLU(True),
            nn.Conv2d(self.DIM, 2 * self.DIM, 5, stride=2, padding=2),
            nn.ReLU(True),
            nn.Conv2d(2 * self.DIM, 4 * self.DIM, 5, stride=2, padding=2),
            nn.ReLU(True),
        )
        self.main = main
        self.output = nn.Linear(4 * 4 * 4 * self.DIM, 1)

    def forward(self, input):
        input = input.view(-1, 1, 207)
        input = self.preprocess(input)
        input = input.view(-1, 1, 16, 16)
        net = self.main(input)
        net = net.view(-1, 2 * 2 * 4 * self.DIM)
        net = self.output(net)
        return net.view(-1)

