import argparse


def get_phaser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=210125,
                        help='random seed')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--model', type=str, default="brits_i")
    parser.add_argument('--hid_size', type=int, default=108)
    parser.add_argument('--impute_weight', type=float, default=0.3)
    parser.add_argument('--label_weight', type=float, default=1.0)
    parser.add_argument('--seq_len', type=int, default=8)
    parser.add_argument('--element_num', type=int, default=1024)

    parser.add_argument('--num-workers', type=int, default=0,
                        help='number workers for dataloader')
    parser.add_argument('--missing-type',  default="random",
                        help='missing type')
    parser.add_argument('--missing-rate',  type=float, default=0.25,
                        help='missing rate')

    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--min-lr', type=float, default=-1,
                        help='min learning rate for LR scheduler. '
                             '-1 to disable annealing')


    return parser

