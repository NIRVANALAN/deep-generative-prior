import os
from copy import deepcopy

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.nn as nn
import torchvision
from PIL import Image
from skimage import color
from skimage.measure import compare_psnr, compare_ssim
from torch.autograd import Variable

from torchvision import transforms

import models
import utils
from models.downsampler import Downsampler

import ipdb

from dataset import default_loader
import sys

sys.path.append("./")

# prepare arguments and save in config
parser = utils.prepare_parser()
parser = utils.add_dgp_parser(parser)
parser = utils.add_example_parser(parser)
parser = utils.add_stylegan_parser(parser)
config = vars(parser.parse_args())
utils.dgp_update_config(config)

# set random seed
utils.seed_rng(config['seed'])

if not os.path.exists('{}/images'.format(config['exp_path'])):
    os.makedirs('{}/images'.format(config['exp_path']))
if not os.path.exists('{}/images_sheet'.format(config['exp_path'])):
    os.makedirs('{}/images_sheet'.format(config['exp_path']))

# create model
generator, discriminator = models.get_model(config['arch'])

G = generator(**config).cuda()
D = discriminator(**config).cuda() if config['ftr_type'] == 'Discriminator' else None

transform = transforms.Compose(
    [utils.CenterCropLongEdge(),
     transforms.Resize(config['resolution']),
     transforms.ToTensor()])

# img_path = 'experiments/inference/test_on_gt/images/000025_inference_not_update_G.png'
# -0.6727
img_path = 'experiments/inference/test_on_gt/images/000025_inference_not_update_G_target.png'
# -1.09
img = transform(default_loader(img_path)).unsqueeze(0).cuda()

ipdb.set_trace()
print(D(img)[0])
