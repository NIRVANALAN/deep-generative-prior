import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.nn as nn
import torchvision
from torchvision.utils import save_image
from PIL import Image
from skimage import color
from skimage.measure import compare_psnr, compare_ssim
from torch.autograd import Variable

import models
import utils
from models.downsampler import Downsampler
import ipdb


class DGP(object):

    def __init__(self, config):
        self.rank, self.world_size = 0, 1
        if config['dist']:
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        self.config = config
        self.mode = config['dgp_mode']
        self.update_G = config['update_G']
        self.update_embed = config['update_embed']  # TODO
        self.iterations = config['iterations']
        self.ftr_num = config['ftr_num']
        self.ft_num = config['ft_num']
        self.lr_ratio = config['lr_ratio']
        self.G_lrs = config['G_lrs']
        self.z_lrs = config['z_lrs']
        self.use_in = config['use_in']
        self.use_D = config['use_D']
        self.select_num = config['select_num']
        self.factor = 2 if self.mode == 'hybrid' else 4  # Downsample factor
        self.arch = config['arch']
        self.verbose = config['verbose']

        # create model
        generator, discriminator = models.get_model(self.arch)

        self.G = generator(**config).cuda()
        self.D = discriminator(**config).cuda() if config['ftr_type'] == 'Discriminator' else None

        model_size = len(self.G.blocks) + 1 if self.arch == 'biggan' else self.G.log_size
        self.G.optim = torch.optim.Adam([{
            'params': self.G.get_params(i, self.update_embed)
        } for i in range(model_size)],
                                        lr=config['G_lr'],
                                        betas=(config['G_B1'], config['G_B2']),
                                        weight_decay=0,
                                        eps=1e-8)

        # load weights
        if config['pose_aware']:
            self.pose_aware_net = {
                'biggan': PoseAwareNet_ZSpace,
                'stylegan': PoseAwareNet_WSpace,
            }[self.arch](config).cuda()

            if config['dgp_mode'] != 'ft':
                # load pose net
                ckpt_pose_path = '{}/pose_aware_net.pth'.format(config['weights_root'])
                print('loading pretrained pose_aware_net from {}'.format(ckpt_pose_path))
                self.pose_aware_net.load_state_dict(
                    torch.load(ckpt_pose_path, map_location=lambda storage, loc: storage))

                # self.pose_aware_net.load_state_dict(
                #     torch.load(ckpt_pose_path, map_location=lambda storage, loc: storage))

        if self.arch == 'biggan':
            utils.load_weights(
                self.G if not (config['use_ema']) else None,
                self.D,
                config['weights_root'],
                name_suffix=config['load_weights'],
                G_ema=self.G if config['use_ema'] else None,
                strict=False)
        elif self.arch == 'stylegan':
            if config['mode'] != 'ft':
                g_weight_path = '{}/G_{}.pth'.format(config['weights_root'], config['load_weights'])
                self.G.load_state_dict(
                    torch.load(g_weight_path, map_location=lambda storage, loc: storage)),
            else:
                ckpt_g = torch.load(config['ckpt_g'], map_location=lambda storage, loc: storage)
                self.G.load_state_dict(ckpt_g["params_ema"])

            ckpt_d = torch.load(config['ckpt_d'], map_location=lambda storage, loc: storage)
            self.D.load_state_dict(ckpt_d["params"])

        else:
            raise NotImplementedError

        self.G.eval()
        if self.D is not None:
            self.D.eval()
        self.G_weight = deepcopy(self.G.state_dict())

        # prepare latent variable and optimizer
        self._prepare_latent(config)
        # prepare learning rate scheduler
        self.G_scheduler = utils.LRScheduler(self.G.optim, config['warm_up'])
        self.z_scheduler = utils.LRScheduler(self.z_optim, config['warm_up'])

        # loss functions
        self.mse = torch.nn.MSELoss()
        if config['ftr_type'] == 'Discriminator':
            self.ftr_net = self.D
            self.criterion = utils.DiscriminatorLoss(ftr_num=config['ftr_num'][0])
        else:
            vgg = torchvision.models.vgg16(pretrained=True).cuda().eval()
            self.ftr_net = models.subsequence(vgg.features, last_layer='20')
            self.criterion = utils.PerceptLoss()

        # Downsampler for producing low-resolution image
        self.downsampler = Downsampler(
            n_planes=3, factor=self.factor, kernel_type='lanczos2', phase=0.5,
            preserve_size=True).type(torch.cuda.FloatTensor)

    def init_hidden(self):
        # for init z after each iteration, pose_aware_net onuy
        self.z = self.z_cache

    def _prepare_latent(self, config):

        # add pose_aware condition network
        params = None
        if config['pose_aware']:
            # same z for the same identity
            if self.arch == 'stylegan':
                # w+ space for stylegan2. BS * 16 * 512
                self.z = torch.zeros(
                    (1, self.G.num_latent, 512),
                    dtype=torch.float,
                    device='cuda',
                    #  requires_grad=True,
                )
                self.z.uniform_(-1, 1)  # follows image2stylegan init for non-face classes
            elif self.arch == 'biggan':
                self.z = torch.zeros((1, self.G.dim_z)).normal_().cuda()  # * normal, dim_z=128
            # self.z.requires_grad = False  # fix z for the same identity
            # optimize encoder along with z
            params = [{
                'params': self.z,
                'lr': config['z_lrs'][0]
            }, {
                'params': self.pose_aware_net.parameters(),
                'lr': config['encoder_lrs'][0]
            }]

        else:
            if self.arch == 'biggan':
                self.z = torch.zeros(
                    (config['z_size'], self.G.dim_z)).normal_().cuda()  # * normal, dim_z=128
                self.z = Variable(self.z, requires_grad=True)
            else:
                self.z = torch.zeros(
                    (config['z_size'], self.G.num_latent, 512),
                    dtype=torch.float,
                    device='cuda',
                    #  requires_grad=True,
                )
                self.z.uniform_(-1, 1)  # follows image2stylegan init for non-face classes
            params = [self.z]

        # TODO lint code
        if self.arch == 'stylegan':
            params = [self.z]

            # Generate list of noise tensors
            noise = []  # stores all of the noise tensors
            noise_vars = []  # stores the noise tensors that we want to optimize on
            for i in range(1, self.G.num_latent):
                # dimension of the ith noise tensor
                res = (config['z_size'], 1, 2**(i // 2 + 2), 2**(i // 2 + 2))

                if (config['noise_type'] == 'zero'
                        or i in [int(layer) for layer in config['bad_noise_layers'].split('.')]):
                    new_noise = torch.zeros(res, dtype=torch.float, device='cuda')
                    new_noise.requires_grad = False
                elif (config['noise_type'] == 'fixed'):
                    new_noise = torch.randn(res, dtype=torch.float, device='cuda')
                    new_noise.requires_grad = False
                elif (config['noise_type'] == 'trainable'):

                    new_noise = torch.randn(res, dtype=torch.float, device='cuda')
                    if (i < config['num_trainable_noise_layers']):
                        new_noise.requires_grad = True
                        noise_vars.append(new_noise)
                    else:
                        new_noise.requires_grad = False
                else:
                    raise Exception("unknown noise type")

                noise.append(new_noise)

            var_list = [self.z] + noise_vars

            # TODO
            # self.z_optim = SphericalOptimizer(opt_func, var_list, lr=learning_rate)

        self.z.requires_grad = True
        self.z_optim = torch.optim.Adam(
            params,
            lr=self.z_lrs[0],
            betas=(self.config['G_B1'], self.config['G_B2']),
            weight_decay=0,
            eps=1e-8)
        # duplicate y for batch z
        self.y = torch.zeros(config['z_size']).long().cuda()

    def reset_G(self):
        self.G.load_state_dict(self.G_weight, strict=False)
        self.G.reset_in_init()
        if self.config['random_G']:
            self.G.train()
        else:
            self.G.eval()

    def random_G(self):
        self.G.init_weights()

    def set_target(self, target, category, img_path, pose=None):
        self.target_origin = target
        # apply degradation transform to the original image
        self.pose = pose.cuda()
        self.target = self.pre_process(target, True)
        self.y.fill_(category.item())
        self.img_name = img_path[img_path.rfind('/') + 1:img_path.rfind('.')]

    def run(self, save_interval=None, meta_data=None):
        save_imgs = self.target.clone()
        save_imgs2 = save_imgs.cpu().clone()
        loss_dict = {}
        curr_step = 0
        count = 0
        ##

        ##
        for stage, iteration in enumerate(self.iterations):
            # setup the number of features to use in discriminator
            # import ipdb.set_trace()
            self.criterion.set_ftr_num(self.ftr_num[stage])

            for i in range(iteration):
                curr_step += 1
                # setup learning rate
                self.G_scheduler.update(curr_step, self.G_lrs[stage], self.ft_num[stage],
                                        self.lr_ratio[stage])
                self.z_scheduler.update(curr_step, self.z_lrs[stage])

                self.z_optim.zero_grad()
                if self.update_G:
                    self.G.optim.zero_grad()

                if self.pose_aware_net and not self.config['fix_posenet_z']:
                    # detach self.z from history
                    # self.init_hidden()
                    z_prime = self.pose_aware_net(self.z, self.pose)
                else:
                    z_prime = self.z

                if self.arch == 'biggan':
                    x = self.G(z_prime, self.G.shared(self.y), use_in=self.use_in[stage])
                elif self.arch == 'stylegan':
                    x = self.G(z_prime, input_is_latent=True, randomize_noise=True)
                # apply degradation transform
                x_map = self.pre_process(x, False)

                ftr_loss = 0
                # ipdb.set_trace()
                # BS must be divisible by 4
                for idx in range(0, len(x_map), 4):
                    ftr_loss += self.criterion(self.ftr_net, x_map[idx:idx + 4],
                                               self.target[idx:idx + 4])
                for idx in range(len(x_map) - len(x_map) % 4, len(x_map)):
                    ftr_loss += self.criterion(self.ftr_net, x_map[None, idx], self.target[None,
                                                                                           idx])

                mse_loss = self.mse(x_map, self.target)
                # nll corresponds to a negative log-likelihood loss # TODO
                nll = z_prime**2 / 2

                nll = nll.mean()
                l1_loss = F.l1_loss(x_map, self.target)

                loss = ftr_loss * self.config['w_D_loss'][stage] + \
                       mse_loss * self.config['w_mse'][stage] + \
                       nll * self.config['w_nll']

                if len(self.config['w_l1']) > 1:
                    loss += l1_loss * self.config['w_l1'][stage]

                # These losses are calculated in the [-1,1] image scale
                # We record the rescaled MSE and L1 loss, corresponding to [0,1] image scale
                loss_dict = {
                    'ftr_loss': ftr_loss,
                    'nll': nll,
                    'mse_loss': mse_loss / 4,
                    'l1_loss': l1_loss / 2
                }

                # discriminator loss
                if self.use_D:
                    dist_fake = self.D(x_map, self.y)[0].squeeze()
                    d_loss = loss_hinge_gen(dist_fake)  # negative
                    # import ipdb
                    # ipdb.set_trace()
                    d_loss *= self.config['w_Disc_loss']
                    loss += d_loss
                    loss_dict.update({'Disc_loss': d_loss})

                loss.backward()

                self.z_optim.step()

                if self.update_G:
                    self.G.optim.step()

                # calculate losses in the non-degradation space
                if self.mode in ['reconstruct', 'colorization', 'SR', 'inpainting']:
                    # x2 is to get the post-processed result in colorization
                    metrics, x2 = self.get_metrics(x)
                    loss_dict = {**loss_dict, **metrics}

                if i == 0 or (i + 1) % self.config['print_interval'] == 0:
                    if self.rank == 0:
                        log_info = ', '.join(
                            ['Stage: [{0}/{1}]'.format(stage + 1, len(self.iterations))] +
                            ['Iter: [{0}/{1}]'.format(i + 1, iteration)] +
                            ['%s : %+4.4f' % (key, loss_dict[key]) for key in loss_dict])
                        print(log_info)

                        with open(Path(self.config['exp_path']) / 'log.txt', 'a+') as f:
                            f.write(log_info + '\n')

                    # save image sheet of the reconstruction process
                    save_imgs = torch.cat((save_imgs, x), dim=0)
                    torchvision.utils.save_image(
                        save_imgs.float(),
                        '%s/images_sheet/%s_%s.jpg' %
                        (self.config['exp_path'], self.img_name, self.mode),
                        nrow=int(save_imgs.size(0)**0.5),
                        normalize=True)
                    if self.mode == 'colorization':
                        save_imgs2 = torch.cat((save_imgs2, x2), dim=0)
                        torchvision.utils.save_image(
                            save_imgs2.float(),
                            '%s/images_sheet/%s_%s_2.jpg' %
                            (self.config['exp_path'], self.img_name, self.mode),
                            nrow=int(save_imgs.size(0)**0.5),
                            normalize=True)

                if save_interval is not None:
                    if i == 0 or (i + 1) % save_interval[stage] == 0:
                        count += 1
                        save_path = '%s/images/%s' % (self.config['exp_path'], self.img_name)
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        img_path = os.path.join(save_path, '%s_%03d.jpg' % (self.img_name, count))
                        utils.save_img(x[0], img_path)

                # stop the reconstruction if the loss reaches a threshold
                if mse_loss.item() < self.config['stop_mse'] or ftr_loss.item(
                ) < self.config['stop_ftr']:
                    break

        # save images
        utils.save_img(
            self.target[0],
            '%s/images/%s_%s_target.png' % (self.config['exp_path'], self.img_name, self.mode))
        utils.save_img(
            self.target_origin[0], '%s/images/%s_%s_target_origin.png' %
            (self.config['exp_path'], self.img_name, self.mode))
        utils.save_img(x[0],
                       '%s/images/%s_%s.png' % (self.config['exp_path'], self.img_name, self.mode))
        if self.mode == 'colorization':
            utils.save_img(
                x2[0], '%s/images/%s_%s2.png' % (self.config['exp_path'], self.img_name, self.mode))

        if self.mode == 'jitter':
            # conduct random jittering
            self.jitter(x)

        if self.config['save_G']:
            torch.save(self.G.state_dict(),
                       '%s/G_%s_%s.pth' % (self.config['exp_path'], self.img_name, self.mode))
            torch.save(self.z,
                       '%s/z_%s_%s.pth' % (self.config['exp_path'], self.img_name, self.mode))
        return loss_dict

    def select_z(self, select_y=False, load_z_path=''):
        with torch.no_grad():
            if self.select_num == 0:
                self.z.zero_()
                return
            elif self.select_num == 1:
                self.z.normal_()
                return
            z_all, y_all, loss_all = [], [], []
            self.criterion.set_ftr_num(3)
            if self.rank == 0:
                if load_z_path != '':
                    print('loading initialized z from {}'.format(load_z_path))
                else:
                    print('Selecting z from {} samples'.format(self.select_num))

            if load_z_path != '':
                # todo
                z_init = torch.load(load_z_path).cuda()
                if z_init.size(0) == 1:
                    return

                if self.target.size(0) == 1:  # one img at a time
                    for z in z_init[:, None]:

                        if self.arch == 'biggan':
                            x = self.G(
                                z,
                                self.G.shared(self.y),
                            )
                        elif self.arch == 'stylegan':
                            x = self.G(z, input_is_latent=True, randomize_noise=True)

                        x = self.pre_process(x)
                        z_all.append(z.cpu())
                        ftr_loss = self.criterion(self.ftr_net, x, self.target)
                        loss_all.append(ftr_loss.view(1).cpu())
                else:
                    for idx in range(self.target.size(0)):
                        for z in z_init[:, None]:
                            if self.arch == 'biggan':
                                x = self.G(
                                    z,
                                    self.G.shared(self.y),
                                )
                            elif self.arch == 'stylegan':
                                x = self.G(z, input_is_latent=True, randomize_noise=True)

                            x = self.pre_process(x)
                            z_all.append(z.cpu())
                            ftr_loss = self.criterion(self.ftr_net, x, self.target[None, idx])
                            loss_all.append(ftr_loss.view(1).cpu())

                        loss_all = torch.cat(loss_all)
                        idx_minloss = torch.argmin(loss_all)
                        self.z[None, idx].copy_(z_all[idx_minloss])
                        z_all, y_all, loss_all = [], [], []

                    return

            else:
                for i in range(self.select_num):
                    self.z.normal_(mean=0, std=self.config['sample_std'])
                    #                    if self.pose_aware_net:
                    #                        self.z = self.pose_aware_net(self.z, self.meta_data['pose'])
                    z_all.append(self.z.cpu())
                    if select_y:
                        self.y.random_(0, self.config['n_classes'])
                        y_all.append(self.y.cpu())
                    if self.arch == 'biggan':
                        # needed?
                        if self.z.size(0) != self.y.size(0):
                            z = self.z.repeat(self.y.size(0), 1)
                        else:
                            z = self.z
                        x = self.G(z, self.G.shared(self.y))
                    elif self.arch == 'stylegan':
                        x = self.G(self.z, input_is_latent=True, randomize_noise=True)
                    else:
                        raise NotImplementedError
                    x = self.pre_process(x)
                    ftr_loss = self.criterion(self.ftr_net, x, self.target)
                    loss_all.append(ftr_loss.view(1).cpu())
                    if self.rank == 0 and (i + 1) % 100 == 0:
                        print('Generating {}th sample'.format(i + 1))

            loss_all = torch.cat(loss_all)
            idx = torch.argmin(loss_all)
            self.z.copy_(z_all[idx])

            if select_y:
                self.y.copy_(y_all[idx])
            self.criterion.set_ftr_num(self.ftr_num[0])

    def pre_process(self, image, target=True):
        if self.mode in ['SR', 'hybrid']:
            # apply downsampling, this part is the same as deep image prior
            if target:
                image_pil = utils.np_to_pil(utils.torch_to_np((image.cpu() + 1) / 2))
                LR_size = [image_pil.size[0] // self.factor, image_pil.size[1] // self.factor]
                img_LR_pil = image_pil.resize(LR_size, Image.ANTIALIAS)
                image = utils.np_to_torch(utils.pil_to_np(img_LR_pil)).cuda()
                image = image * 2 - 1
            else:
                image = self.downsampler((image + 1) / 2)
                image = image * 2 - 1
            # interpolate to the orginal resolution via bilinear interpolation
            image = F.interpolate(image, scale_factor=self.factor, mode='bilinear')
        n, _, h, w = image.size()
        if self.mode in ['colorization', 'hybrid']:
            # transform the image to gray-scale
            r = image[:, 0, :, :]
            g = image[:, 1, :, :]
            b = image[:, 2, :, :]
            gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
            image = gray.view(n, 1, h, w).expand(n, 3, h, w)
        if self.mode in ['inpainting', 'hybrid']:
            # remove the center part of the image
            hole = min(h, w) // 3
            begin = (h - hole) // 2
            end = h - begin
            self.begin, self.end = begin, end
            mask = torch.ones(1, 1, h, w).cuda()
            mask[0, 0, begin:end, begin:end].zero_()
            image = image * mask
        return image

    def get_metrics(self, x):
        with torch.no_grad():
            l1_loss_origin = F.l1_loss(x, self.target_origin) / 2
            mse_loss_origin = self.mse(x, self.target_origin) / 4
            metrics = {'l1_loss_origin': l1_loss_origin, 'mse_loss_origin': mse_loss_origin}
            # transfer to numpy array and scale to [0, 1]
            target_np = (self.target_origin.detach().cpu().numpy()[0] + 1) / 2
            x_np = (x.detach().cpu().numpy()[0] + 1) / 2
            target_np = np.transpose(target_np, (1, 2, 0))
            x_np = np.transpose(x_np, (1, 2, 0))
            if self.mode == 'colorization':
                # combine the 'ab' dim of x with the 'L' dim of target image
                x_lab = color.rgb2lab(x_np)
                target_lab = color.rgb2lab(target_np)
                x_lab[:, :, 0] = target_lab[:, :, 0]
                x_np = color.lab2rgb(x_lab)
                x = torch.Tensor(np.transpose(x_np, (2, 0, 1))) * 2 - 1
                x = x.unsqueeze(0)
            elif self.mode == 'inpainting':
                # only use the inpainted area to calculate ssim and psnr
                x_np = x_np[self.begin:self.end, self.begin:self.end, :]
                target_np = target_np[self.begin:self.end, self.begin:self.end, :]
            ssim = compare_ssim(target_np, x_np, multichannel=True)
            psnr = compare_psnr(target_np, x_np)
            metrics['psnr'] = torch.Tensor([psnr]).cuda()
            metrics['ssim'] = torch.Tensor([ssim]).cuda()
            return metrics, x

    def jitter(self, x):
        save_imgs = x.clone().cpu()
        z_rand = self.z.clone()
        stds = [0.3, 0.5, 0.7]
        save_path = '%s/images/%s_jitter' % (self.config['exp_path'], self.img_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with torch.no_grad():
            for std in stds:
                for i in range(30):
                    # add random noise to the latent vector
                    z_rand.normal_()
                    z = self.z + std * z_rand
                    if self.arch == 'biggan':
                        x_jitter = self.G(z, self.G.shared(self.y))
                    elif self.arch == 'stylegan':
                        x_jitter = self.G([self.z])
                    utils.save_img(x_jitter[0], '%s/std%.1f_%d.jpg' % (save_path, std, i))
                    save_imgs = torch.cat((save_imgs, x_jitter.cpu()), dim=0)

        torchvision.utils.save_image(
            save_imgs.float(),
            '%s/images_sheet/%s_jitters.jpg' % (self.config['exp_path'], self.img_name),
            nrow=int(save_imgs.size(0)**0.5),
            normalize=True)


class PoseAwareNet_ZSpace(nn.Module):

    def __init__(self, config, dim_z=119):
        # self.pose = config['pose']
        super(PoseAwareNet_ZSpace, self).__init__()
        # E_Linear in stylegan?
        self.network = nn.Sequential(
            nn.Linear(dim_z + config['pose_dim'], dim_z),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(dim_z, dim_z),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(dim_z, dim_z),
            nn.LeakyReLU(negative_slope=0.2),
        )

    def forward(self, z, pose):
        if z.size(0) != pose.size(0):
            assert z.size(0) == 1
            # repeat for the same identity code
            z = z.repeat(pose.size(0), 1)
        z = torch.cat([z, pose], dim=1)
        z = self.network(z)
        return z


class lift_MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        self.fusion_net = [
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(hidden_dim, output_dim),
            nn.LeakyReLU(negative_slope=0.2),
        ]
        self.fusion_net = nn.Sequential(*self.fusion_net)

    def forward(self, x):
        return self.fusion_net(x)


class PoseAwareNet_WSpace(nn.Module):

    # like encoder in styleRig
    def __init__(self, config, dim_z=512, num_rig_layers=4):
        # self.pose = config['pose']
        super(PoseAwareNet_WSpace, self).__init__()
        for i in range(num_rig_layers):
            self.__setattr__('mlp_{}'.format(i), lift_MLP(config['pose_dim'], 128, dim_z))
        self.num_rig_layers = num_rig_layers

    def forward(self, w, pose):

        if w.size(0) != pose.size(0):
            assert w.size(0) == 1
            # repeat for the same identity code
            w = w.repeat(pose.size(0), 1, 1)
        delta_w = [
            getattr(self, 'mlp_{}'.format(i))(pose.unsqueeze(1)) for i in range(self.num_rig_layers)
        ]
        for i in range(self.num_rig_layers, w.size(1)):
            delta_w.append(torch.zeros_like(delta_w[0]))
        delta_w = torch.cat(delta_w, 1)

        # only add to first 4 layers of W space
        w += delta_w

        return w


# hinge loss used in BigGAN
def loss_hinge_dis(dis_flag, dis_score):
    # dis_fake, dis_real):
    assert dis_flag in ['real', 'fake']
    if dis_flag == 'real':
        return torch.mean(F.relu(1. - dis_score))
    elif dis_flag == 'fake':
        return torch.mean(F.relu(1. + dis_score))


# hinge loss for Generator
def loss_hinge_gen(dis_fake):
    loss = -torch.mean(dis_fake)
    return loss
