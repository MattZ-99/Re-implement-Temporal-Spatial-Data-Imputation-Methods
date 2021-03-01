import argparse


def get_phaser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=3,
                        help='random seed')
    # training options
    parser.add_argument('--plot-interval', type=int, default=50,
                        help='plot interval. 0 to disable plotting.')
    parser.add_argument('--save-interval', type=int, default=0,
                        help='interval to save models. 0 to disable saving.')
    parser.add_argument('--mask', default='block',
                        help='missing data mask. (options: block, indep)')
    # option for block: set to 0 for variable size
    parser.add_argument('--block-len', type=int, default=12,
                        help='size of observed block')
    parser.add_argument('--block-len-max', type=int, default=None,
                        help='max size of observed block. '
                             'Use fixed-size observed block if unspecified.')
    # option for indep:
    parser.add_argument('--obs-prob', type=float, default=.2,
                        help='observed probability for independent dropout')
    parser.add_argument('--obs-prob-max', type=float, default=None,
                        help='max observed probability for independent '
                             'dropout. Use fixed probability if unspecified.')

    parser.add_argument('--flow', type=int, default=2,
                        help='number of IAF layers')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--min-lr', type=float, default=-1,
                        help='min learning rate for LR scheduler. '
                             '-1 to disable annealing')

    parser.add_argument('--arch', default='conv',
                        help='network architecture. (options: fc, conv)')
    parser.add_argument('--epoch', type=int, default=1000,
                        help='number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--ae', type=float, default=.1,
                        help='autoencoding regularization strength')
    parser.add_argument('--prefix', default='pbigan',
                        help='prefix of output directory')
    parser.add_argument('--latent', type=int, default=128,
                        help='dimension of latent variable')
    parser.add_argument('--aeloss', default='bce',
                        help='autoencoding loss. '
                             '(options: mse, bce, smooth_l1, l1)')
    parser.add_argument('--time-stamp-length', type=int, default=8,
                        help='time-stamp-length')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='number workers for dataloader')
    parser.add_argument('--missing-type',  default="random",
                        help='missing type')
    parser.add_argument('--missing-rate',  type=float, default=0.25,
                        help='missing rate')
    return parser

