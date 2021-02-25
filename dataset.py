import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

from pathlib import Path
import glob

import utils
import torch
import numpy as np
from pathlib import Path
import ipdb

import quaternion


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ImageDataset(data.Dataset):

    def __init__(self, root_dir, meta_file=None, transform=None, image_size=128, normalize=True):
        self.root_dir = root_dir
        if transform is not None:
            self.transform = transform
        else:
            norm_mean = [0.5, 0.5, 0.5]
            norm_std = [0.5, 0.5, 0.5]
            if normalize:
                self.transform = transforms.Compose([
                    utils.CenterCropLongEdge(),
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(norm_mean, norm_std)
                ])
            else:
                self.transform = transforms.Compose([
                    utils.CenterCropLongEdge(),
                    transforms.Resize(image_size),
                    transforms.ToTensor()
                ])
        if meta_file is not None:
            with open(meta_file) as f:
                lines = f.readlines()
        else:
            lines = glob.glob(Path(root_dir) / '*.png')

        print("building dataset from %s" % meta_file)
        self.num = len(lines)
        self.metas = []
        self.classifier = None
        for line in lines:
            line_split = line.rstrip().split()
            if len(line_split) == 2:
                self.metas.append((line_split[0], int(line_split[1])))
            else:
                self.metas.append((line_split[0], -1))
        print("read meta done")

    def __len__(self):
        return self.num

    def read_file(self, idx):
        filename = self.root_dir + '/' + self.metas[idx][0]
        cls = self.metas[idx][1]
        img = default_loader(filename)
        return filename, cls, img

    def __getitem__(self, idx):
        filename, cls, img = self.read_file(idx)
        # transform
        if self.transform is not None:
            img = self.transform(img)

        return img, cls, self.metas[idx][0]


class ImagePoseDataset(ImageDataset):

    def __init__(self,
                 root_dir,
                 meta_file=None,
                 transform=None,
                 image_size=128,
                 normalize=True,
                 pose_format='quaternion'):
        super().__init__(root_dir, meta_file, transform, image_size, normalize)
        assert pose_format in ['quaternion', 'euler']
        self.pose_format = pose_format

    def read_pose(self, idx):
        pose_path = Path(self.root_dir).parent / 'pose' / '{:06}.txt'.format(
            int(self.metas[idx][0].split('.')[0]))
        pose = torch.from_numpy(np.loadtxt(pose_path, dtype=np.float32).reshape(4, 4))[:3, :]
        rotation_mat = pose[:3, :3]
        camera_pos = pose[:3, 3].view(-1)

        if self.pose_format == 'quaternion':
            pose = torch.from_numpy(
                quaternion.as_float_array(quaternion.from_rotation_matrix(rotation_mat)))
        else:
            # TODO
            pass
        viewpoint = torch.cat([camera_pos, pose]).float()

        return viewpoint

    def __getitem__(self, idx):
        img, cls, img_path = super().__getitem__(idx=idx)
        meta_data = {'image': img, 'img_path': img_path, 'cls': cls, 'pose': self.read_pose(idx)}

        return meta_data
