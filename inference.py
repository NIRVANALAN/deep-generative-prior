import os
import sys
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from collections import OrderedDict

import torch
import torchvision.utils as vutils
import torch.optim
from torch.utils.data import DataLoader

from dataset import ImageDataset
import utils
from models import DGP

import ipdb

sys.path.append("./")


# Arguments for demo
def add_example_parser(parser):
    parser.add_argument(
        '--image_path',
        type=str,
        default='',
        help='Path of the image to be processed (default: %(default)s)')
    parser.add_argument(
        '--class', type=int, default=-1, help='class index of the image (default: %(default)s)')
    parser.add_argument(
        '--image_path2',
        type=str,
        default='',
        help='Path of the 2nd image to be processed, used in "morphing" mode (default: %(default)s)'
    )
    parser.add_argument(
        '--class2',
        type=int,
        default=-1,
        help='class index of the 2nd image, used in "morphing" mode (default: %(default)s)')
    parser.add_argument(
        '--reset_G', action='store_true', default=False, help='reset G after each epoch')

    return parser


# prepare arguments and save in config
parser = utils.prepare_parser()
parser = utils.add_dgp_parser(parser)
parser = add_example_parser(parser)
config = vars(parser.parse_args())
utils.dgp_update_config(config)

# set random seed
utils.seed_rng(config['seed'])

if not os.path.exists('{}/images'.format(config['exp_path'])):
    os.makedirs('{}/images'.format(config['exp_path']))
if not os.path.exists('{}/images_sheet'.format(config['exp_path'])):
    os.makedirs('{}/images_sheet'.format(config['exp_path']))

# initialize DGP model
dgp = DGP(config)

# prepare dataset loader

train_dataset = ImageDataset(
    config['root_dir'], config['list_file'], image_size=config['resolution'], normalize=True)
sampler = utils.DistributedSampler(train_dataset) if config['dist'] else None

train_loader = DataLoader(
    train_dataset, batch_size=1, shuffle=False, sampler=sampler, num_workers=1, pin_memory=False)

# cls_category = torch.Tensor([config['class']]).long()

cls_category = None

# loss_dict = dgp.run()

model = dgp

# core optm loop
for i, (image, category, img_path) in enumerate(train_loader):
    # measure data loading time
    torch.cuda.empty_cache()

    cls_category = category.item()

    image = image.cuda()
    category = category.cuda()
    img_path = img_path[0]

    # prepare initial latent vector
    if not config['update_G']:
        model.reset_G()  #*

    model.set_target(image, category, img_path)
    # when category is unkonwn (category=-1), it would be selected from samples
    model.select_z(select_y=True if category.item() < 0 else False)
    # start reconstruction
    loss_dict = model.run(save_interval=config['save_interval'])

ipdb.set_trace()

print(cls_category)

torch.save(model.G.state_dict(),
           '%s/G_%s_%s.pth' % (config['exp_path'], category.item(), config['dgp_mode']))
torch.save(model.z, '%s/z_%s_%s.pth' % (config['exp_path'], category.item(), config['dgp_mode']))

# 1. fine tuned G. 2. output directory