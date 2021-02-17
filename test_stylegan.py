import os
import sys
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from collections import OrderedDict

import torch

from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils
from tqdm import tqdm
import torchvision.utils as vutils

import utils
from models import StyleGANGenerator

# from models import StyleGANGenerator, StyleGANDiscriminator

sys.path.append("./")

# parser = argparse.ArgumentParser(description="StyleGAN2 trainer")

# prepare arguments and save in config
parser = utils.prepare_parser()
parser = utils.add_dgp_parser(parser)
parser = utils.add_stylegan_parser(parser)
config = vars(parser.parse_args())
utils.dgp_update_config(config)

# set random seed
utils.seed_rng(config['seed'])

if not os.path.exists('{}/images'.format(config['exp_path'])):
    os.makedirs('{}/images'.format(config['exp_path']))
if not os.path.exists('{}/images_sheet'.format(config['exp_path'])):
    os.makedirs('{}/images_sheet'.format(config['exp_path']))

args = parser.parse_args()
device = "cuda"

args.latent = 512
args.n_mlp = 8
# args.expname = 'stylegan_sample'
# Path('{}_size{}_bs{}_iter{}'.format( Path(args.path).name, args.size, args.batch, args.iter))

if __name__ == '__main__':

    print("load model:", args.ckpt_g)
    g_ema = StyleGANGenerator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier).to(device)
    g_ema.eval()

    # define model

    ckpt = torch.load(args.ckpt_g, map_location=lambda storage, loc: storage)
    g_ema.load_state_dict(ckpt['params_ema'])

    with torch.no_grad():
        g_ema.eval()
        for i in range(10):
            sample_z = torch.randn(args.n_sample, args.latent, device=device)
            sample = g_ema([sample_z])
            print(sample.size())

            # ipdb.set_trace()

            vutils.save_image(
                sample,
                '{}/images/{}.jpg'.format(config['exp_path'], i),
                nrow=int(args.n_sample**0.5),
                normalize=True,
                range=(-1, 1),
            )
