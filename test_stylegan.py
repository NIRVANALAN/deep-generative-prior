import os
import sys
import math
import numpy as np
from pathlib import Path

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from collections import OrderedDict

import torch
import ipdb

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
args.size = 512
args.log_size = int(math.log(args.size, 2))
args.n_sample = 16

# args.num_layers = args.log_size *
# args.expname = 'stylegan_sample'
# Path('{}_size{}_bs{}_iter{}'.format( Path(args.path).name, args.size, args.batch, args.iter))


def load_latent(latent_in_path):
    if Path(latent_in_path).exists():
        try:
            latent = torch.load(latent_in_path)
        except:
            latent = torch.from_numpy(np.load(latent_in_path)).cuda()
    else:
        print("\tRunning Mapping Network")
        with torch.no_grad():
            # torch.manual_seed(0)
            latent = torch.randn((100, 512), dtype=torch.float32, device="cuda")
            torch.save(latent, latent_in_path)
    return latent


def sample_stylegan(g_ema,
                    args,
                    identity_idx=None,
                    style_indices=None,
                    latent_code=None,
                    input_is_latent=False,
                    inject_index=None,
                    iter=1,
                    name_prefix='',
                    cross_over=False):
    """[sample from stylegan]

    Args:
        g_ema (nn.Module): trained stylegan ema generator
        args (dict): args
        content_idx (int, optional): latent code idx for identity. Defaults to None.
        style_idx (int, optional): latent code idx for pose. Defaults to None.
        latent_code (torch.Tensor, optional): sampled standard gaussian prior z code. Defaults to None.
        input_is_latent (bool, optional): whether input is w space. Defaults to False.
        inject_index (int, optional): which layer in stylegan to perform crossover. Defaults to None.
        iter (int, optional): iterations of sampling. Defaults to 10.
        name_prefix (str, optional): to erase confusion of naming. Defaults to ''.
    """
    with torch.no_grad():
        g_ema.eval()
        if not os.path.exists('{}/images/{}'.format(config['exp_path'], inject_index)):
            os.makedirs('{}/images/{}'.format(config['exp_path'], inject_index))
        for i in tqdm(range(iter)):
            if latent_code is None:
                latent_code = torch.randn(args.n_sample, args.latent, device=device)
            if isinstance(latent_code, torch.Tensor) and latent_code.dim() == 1:
                latent_code = latent_code.unsqueeze(0)

            if cross_over:
                for style_idx in tqdm(style_indices):
                    latent_code_pair = latent_code[[style_idx, identity_idx]].unsqueeze(1).unbind(0)

                    # import ipdb
                    # ipdb.set_trace()

                    sample = g_ema(
                        latent_code_pair,
                        input_is_latent=input_is_latent,
                        inject_index=inject_index)
                    vutils.save_image(
                        sample,
                        '{}/images/crossover/src{}_style_{}_inject{}.jpg'.format(
                            config['exp_path'],
                            identity_idx,
                            style_idx,
                            inject_index,
                        ),
                        nrow=int(args.n_sample**0.5),
                        normalize=True,
                        range=(-1, 1),
                    )
            else:
                sample = g_ema(latent_code, input_is_latent=input_is_latent)
                # ipdb.set_trace()
                vutils.save_image(
                    sample,
                    '{}/images/{}_sample.jpg'.format(
                        config['exp_path'],
                        i,
                    ),
                    nrow=int(args.n_sample**0.5),
                    normalize=True,
                    range=(-1, 1),
                )


if __name__ == '__main__':

    print("load model:", args.ckpt_g)
    g_ema = StyleGANGenerator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier).to(device)
    g_ema.eval()

    # define model

    ckpt = torch.load(args.ckpt_g, map_location=lambda storage, loc: storage)
    g_ema.load_state_dict(ckpt['params_ema'])

    # calculate statistics of mapping network
    mapping = g_ema.style_mlp
    lrelu = torch.nn.LeakyReLU(negative_slope=0.2)
    gaussian_fit_path = 'gaussian_fit_lsun.pt'
    # latent_in_path = 'stylegan2_lsun_latent.pt'
    latent_1 = '/mnt/lustre/yslan/Repo/Generation/Inversion/StyleGAN_LatentEditor/latent_W/000000.pt'
    latent_2 = '/mnt/lustre/yslan/Repo/Generation/Inversion/StyleGAN_LatentEditor/latent_W/000100.npy'

    latent = torch.cat([load_latent(latent_path) for latent_path in [latent_1, latent_2]], dim=0)
    # latent = load_latent(latent_in_path)
    print('latent shape: {}'.format(latent.shape))

    # if Path(gaussian_fit_path).exists():
    #     gaussian_fit = torch.load(gaussian_fit_path)

    # prepare latent

    # else:
    #     latent = torch.randn((args.n_sample, 16, 512),
    #                          dtype=torch.float,
    #                          requires_grad=True,
    #                          device='cuda')
    #     torch.save(latent, latent_in_path)
    #     print("\tSaved {}".format(latent_in_path))

    # latent_in = lrelu(latent * gaussian_fit["std"] + gaussian_fit["mean"])

    # with torch.no_grad():
    #     g_ema.eval()
    #     for i in range(1):
    #         sample_z = torch.randn(args.n_sample, args.latent, device=device)
    #         import ipdb
    #         ipdb.set_trace()
    #         sample = g_ema([sample_z])
    #         print(sample.size())

    #         vutils.save_image(
    #             sample,
    #             '{}/images/{}.jpg'.format(config['exp_path'], i),
    #             nrow=int(args.n_sample**0.5),
    #             normalize=True,
    #             -range=(-1, 1),
    #         )

    #inference from given latent_code

    # sample_stylegan(
    #     g_ema,
    #     args,
    #     # latent_code=latent[[0, 1]].unsqueeze(1).unbind(0),
    #     # latent_code=latent[1],
    #     latent_code=latent[None, 1],
    #     # latent_code=latent,
    #     identity_idx=11,
    #     style_indices=range(50),
    #     # input_is_latent=False,
    #     input_is_latent=True,
    #     cross_over=False,
    #     inject_index=4,
    #     iter=1,
    # )

    # perform crossover
    # '''
    sample_stylegan(
        g_ema,
        args,
        latent_code=latent[[0, 1]],
        # latent_code=latent,
        # latent_code=latent[0].unsqueeze(0),
        # latent_code=latent,
        identity_idx=1,
        style_indices=[0],
        input_is_latent=True,
        cross_over=True,
        inject_index=8,
        iter=1,
    )
    # '''