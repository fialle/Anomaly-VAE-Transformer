import os
import argparse

from torch.backends import cudnn
from solver import Solver


def str2bool(v):
    return v.lower() in ('true')


def main(config):
    if (not os.path.exists(config.model_save_path)):
        os.makedirs(config.model_save_path)
    solver = Solver(vars(config))

    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()

    return solver


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--win_size', type=int, default=28)
    parser.add_argument('--input_c', type=int, default=64)
    parser.add_argument('--output_c', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--dataset', type=str, default='Olympus_VAE')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--data_path', type=str, default='./datasets')
    parser.add_argument('--model_save_path', type=str, default='checkpoints')
    parser.add_argument('--latent_dim', type=int, default=64)
    parser.add_argument('--num_hidden_layers', type=int, default=2)

    config = parser.parse_args()

    #config.mode = 'train'
    #config.mode = 'test'

    args = vars(config)
    main(config)
