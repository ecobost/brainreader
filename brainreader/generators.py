""" Generators for MNIST images and natural images.

MNIST: Generator from a simple DCGAN or an MLP VAE: 
    https://github.com/csinva/gan-vae-pretrained-pytorch

ImageNet: BigGAN
"""
from torch import nn
import torch
from os import path
from torch.nn import functional as F

mnist_gan = 'https://github.com/csinva/gan-vae-pretrained-pytorch/raw/master/mnist_dcgan/weights/netG_epoch_99.pth'
mnist_vae = 'https://github.com/csinva/gan-vae-pretrained-pytorch/raw/master/mnist_vae/weights/vae_epoch_25.pth'


class Generator(nn.Module):
    def __init__(self, nc=1, nz=100, ngf=64):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, kernel_size=1, stride=1, padding=2, bias=False),
            nn.Tanh())

    def forward(self, input):
        output = (self.main(input) + 1) / 2 # [0, 1 range]
        return output


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    # def forward(self, x):
    #     mu, logvar = self.encode(x.reshape(-1, 784))
    #     z = self.reparameterize(mu, logvar)
    #     return self.decode(z), mu, logvar

    def forward(self, z):
        return self.decode(z).view(-1, 1, 28, 28)


def download_file(url, filename):
    """ Downloads a file from the provided url in path. """
    # Download the weights (if needed)
    if not path.exists(filename):
        from urllib import request

        print(f'Downloading weights to {filename}...')
        request.urlretrieve(url, filename=filename)


def get_mnist_gan(download_path='/mnt/scratch07/ecobost/generators/netG_epoch_99.pth',
                  num_gpus=1):
    """ Loads and returns a pretrained generator. 
    
    Arguments:
        download_path (string): Where the weights will be downloaded (if not already 
            present).
    
    Returns:
        The generator. A pytorch module
    """
    # Download the weights (if needed)
    download_file(mnist_gan, download_path)

    # Create generator
    generator = Generator()
    generator.load_state_dict(torch.load(download_path))
    return generator


def get_mnist_vae(download_path='/mnt/scratch07/ecobost/generators/vae_epoch_25.pth'):
    """ Loads and returns a pretrained generator. 
    
    Arguments:
        download_path (string): Where the weights will be downloaded (if not already 
            present).
    
    Returns:
        The generator. A pytorch module
    """
    # Download the weights (if needed)
    download_file(mnist_vae, download_path)

    # Create generator
    generator = VAE()
    generator.load_state_dict(torch.load(download_path))
    return generator


##########################################################################################
""" Pretrained BigGAN model from:
    https://github.com/huggingface/pytorch-pretrained-BigGAN
    
Install it as:
    pip3 install pytorch-pretrained-biggan
"""

# from pytorch_pretrained_biggan import BigGAN

# class BigGANWrapper(BigGAN):
#     """ Wraps the original BigGAN module to modify the forward. """
#     def forward(self, z):
#         """ Modified forward to work on a single 256-dim vector in N(0, 1)."""
#         truncated_noise = torch.tanh(z[:, :128]) * self._truncation * 2 # this 2 
#         class_vector = z[:, 128:] * 0.06 # embeddings have ~this std
#         new_z = torch.cat((truncated_noise, class_vector), dim=1)
#         return self.generator(new_z, self._truncation)

# def get_biggan(cache_dir='/mnt/scratch07/ecobost/generators/', truncation=0.6):
#     """ Get the pretrained BigGAN.
    
#     Arguments:
#         cache_dir (string): Where to save the model
#         truncation (float): Between (0, 1], how to truncate the noise values. Bigger 
#             values produce more diverse (but worse looking) samples. Value of 1 actually
#             truncates the normal at (-2, 2) (odd but that's how it's been defined).
#     """
#     model = BigGANWrapper.from_pretrained('biggan-deep-256', cache_dir=cache_dir)
#     model._truncation = truncation
#     return model