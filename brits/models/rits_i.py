import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable
from torch.nn.parameter import Parameter

import math

from ipdb import set_trace
from sklearn import metrics


def binary_cross_entropy_with_logits(input, target, weight=None, size_average=True, reduce=True):
    if not (target.size() == input.size()):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

    max_val = (-input).clamp(min=0)
    loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()

    if weight is not None:
        loss = loss * weight

    if not reduce:
        return loss
    elif size_average:
        return loss.mean()
    else:
        return loss.sum()


class TemporalDecay(nn.Module):
    def __init__(self, input_size, rnn_hid_size):
        super(TemporalDecay, self).__init__()
        self.rnn_hid_size = rnn_hid_size
        self.build(input_size)

    def build(self, input_size):
        self.W = Parameter(torch.Tensor(self.rnn_hid_size, input_size))
        self.b = Parameter(torch.Tensor(self.rnn_hid_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, d):
        gamma = F.relu(F.linear(d, self.W, self.b))
        gamma = torch.exp(-gamma)
        return gamma


class Model(nn.Module):
    def __init__(self, rnn_hid_size, seq_len, element_num):
        super(Model, self).__init__()

        self.rnn_hid_size = rnn_hid_size
        self.seq_len = seq_len
        self.input_size = element_num

        self.out = nn.Linear(self.rnn_hid_size, 1)
        self.temp_decay = TemporalDecay(input_size=self.input_size, rnn_hid_size=self.rnn_hid_size)
        self.regression = nn.Linear(self.rnn_hid_size, self.input_size)
        self.rnn_cell = nn.LSTMCell(self.input_size * 2, self.rnn_hid_size)

    def forward(self, data, direct):

        values = data[direct]['values']
        masks = data[direct]['masks']
        deltas = data[direct]['deltas']

        h = torch.zeros((values.size()[0], self.rnn_hid_size))
        h.requires_grad = True
        c = torch.zeros((values.size()[0], self.rnn_hid_size))
        c.requires_grad = True

        if torch.cuda.is_available():
            h, c = h.cuda(), c.cuda()

        x_loss = 0.0

        imputations = []

        for t in range(self.seq_len):
            x = values[:, t, :]
            m = masks[:, t, :]
            d = deltas[:, t, :]

            gamma = self.temp_decay(d)
            h = h * gamma
            x_h = self.regression(h)

            x_c = m * x + (1 - m) * x_h

            x_loss += torch.sum(torch.abs(x - x_h) * m) / (torch.sum(m) + 1e-5)

            inputs = torch.cat([x_c, m], dim=1)

            h, c = self.rnn_cell(inputs, (h, c))

            imputations.append(x_c.unsqueeze(dim=1))

        imputations = torch.cat(imputations, dim=1)

        return {'loss': x_loss, 'imputations': imputations}

    def run_on_batch(self, data, optimizer, epoch=None):
        ret = self(data, direct='forward')

        if optimizer is not None:
            optimizer.zero_grad()
            ret['loss'].backward()
            optimizer.step()

        return ret
