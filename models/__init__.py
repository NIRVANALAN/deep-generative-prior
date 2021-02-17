from .biggan import *
from .dgp import *
from .nethook import *

from .dgp_stylegan import DGPStyleGAN

from .stylegan2 import Generator as StyleGANGenerator
from .stylegan2 import Discriminator as StyleGANDiscriminator

# from .stylegan import Generator as StyleGANGenerator
# from .stylegan import Discriminator as StyleGANDiscriminator


def get_model(arch):
    if arch == 'biggan':
        return Generator, Discriminator
    elif arch == 'stylegan':
        return StyleGANGenerator, StyleGANDiscriminator
    else:
        raise NotImplementedError
