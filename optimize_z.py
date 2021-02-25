import os
import sys
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from collections import OrderedDict

import torch
import torchvision.utils as vutils
import torch.optim
from torch.utils.data import DataLoader

from dataset import ImageDataset, ImagePoseDataset
import utils
from models import DGP

import ipdb

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
if not os.path.exists('{}'.format(config['save_weights_dir'])):
    os.makedirs('{}'.format(config['save_weights_dir']))

# prepare dataset loader

# train_dataset = ImageDataset(
train_dataset = ImagePoseDataset(
    config['root_dir'],
    config['list_file'],
    image_size=config['resolution'],
    normalize=True,
    transform=None)

# sampler = utils.DistributedSampler(train_dataset) if config['dist'] else None
# use all samples to finetune
bs = len(train_dataset) if config['mode'] == 'ft' else 1
print(config['mode'])

train_loader = DataLoader(
    train_dataset,
    batch_size=bs,
    shuffle=False,
    # sampler=sampler,
    num_workers=0,
    pin_memory=False)

config['z_size'] = bs

# cls_category = torch.Tensor([config['class']]).long()
cls_category = None
# initialize DGP model
dgp = DGP(config)

# loss_dict = dgp.run()

model = dgp

# core optim loop
# for i, (image, category, img_path) in enumerate(train_loader):
for i, meta_data in enumerate(train_loader):
    # measure data loading time
    torch.cuda.empty_cache()
    # TODO
    image, img_path, category, pose = meta_data['image'], meta_data['img_path'], meta_data[
        'cls'], meta_data['pose']

    image = image.cuda()
    img_path = img_path[0]
    if bs != 1:
        category = category[0]

    category = category.cuda()

    # prepare initial latent vector
    # if not config['update_G']:
    #     model.reset_G()  #*

    model.set_target(image, category, img_path, pose)
    # when category is unknown (category=-1), it would be selected from samples
    model.select_z(
        select_y=True if category.item() < 0 else False, load_z_path=config['load_z_path'])
    # start reconstruction
    loss_dict = model.run(save_interval=config['save_interval'])

torch.save(model.G.state_dict(),
           '%s/G_%s_%s.pth' % (config['save_weights_dir'], category.item(), config['dgp_mode']))
torch.save(model.z,
           '%s/z_%s_%s.pth' % (config['save_weights_dir'], category.item(), config['dgp_mode']))

# 1. fine tuned G. 2. output directory
