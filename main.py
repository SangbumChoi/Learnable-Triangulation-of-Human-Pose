import os
import argparse

import torch

from models.loss import MSEloss, L1loss
from models.algebrictriangulation import algebrictriangulation
from models.volumetrictriangulation import volumetrictriangulation

import cfg

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--eval_dataset', type=str, default='alg')


    args = parser.parse_args()
    return args

def main(args):
    print("Number of available GPUs: {}".format(torch.cuda.device_count()))

    device = torch.device('cpu')
    print("device", device)

    # config
    config = cfg.load_config(args.config)
    config.opt.n_iters_per_epoch = config.opt.n_objects_per_epoch // config.opt.batch_size

    model = {
        "alg": algebrictriangulation,
        "vol": volumetrictriangulation
    }[config.model.name](config, device=device).to(device)

    if config.model.name == "vol":
        criterion = L1loss
    else:
        criterion = MSEloss


if __name__ == '__main__':
    args = parse_args()
    print("args: {}".format(args))
    main(args)