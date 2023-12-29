import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchinfo import summary
from torch.autograd import grad as torch_grad
from torch.linalg import vector_norm as torch_vnorm

from dataset import mnist


λ = 10 # hyperparameter controlling strength of the gradient penalty


def show_img(img):
  plt.imshow(img.detach().numpy().reshape(-1, 28, 28)[0])
  plt.show()


class Discriminator(nn.Module):
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Conv2d( 1, 32, 5),
      nn.LeakyReLU(),
      nn.Conv2d(32, 32, 5),
      nn.LayerNorm([32, 20, 20]),
      nn.LeakyReLU(),
      nn.Conv2d(32, 48, 5),
      nn.LeakyReLU(),
      nn.Conv2d(48, 48, 5),
      nn.LayerNorm([48, 12, 12]),
      nn.LeakyReLU(),
      nn.AvgPool2d(3, 3),
      nn.Flatten(),
      nn.Linear(768, 400),
      nn.LayerNorm([400]),
      nn.LeakyReLU(),
      nn.Linear(400, 400),
      nn.LayerNorm([400]),
      nn.LeakyReLU(),
      nn.Linear(400, 400),
      nn.LayerNorm([400]),
      nn.LeakyReLU(),
      nn.Linear(400, 1, bias=False),
    )
  def forward(self, x):
    return self.layers(x.reshape(-1, 1, 28, 28))


class Generator(nn.Module):
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(128, 400),
      nn.LeakyReLU(),
      nn.Linear(400, 400),
      nn.LayerNorm([400]),
      nn.LeakyReLU(),
      nn.Linear(400, 400),
      nn.LayerNorm([400]),
      nn.LeakyReLU(),
      nn.Linear(400, 768),
      nn.Unflatten(1, [48, 4, 4]),
      nn.ConvTranspose2d(48, 48, 3, 3),
      nn.LayerNorm([48, 12, 12]),
      nn.LeakyReLU(),
      nn.ConvTranspose2d(48, 48, 5),
      nn.LeakyReLU(),
      nn.ConvTranspose2d(48, 32, 5),
      nn.LayerNorm([32, 20, 20]),
      nn.LeakyReLU(),
      nn.ConvTranspose2d(32, 32, 5),
      nn.LeakyReLU(),
      nn.ConvTranspose2d(32, 1, 5),
      nn.Sigmoid(),
    )
  def forward(self, z):
    return self.layers(z).reshape(-1, 28, 28)


class WGANDisc:
  """ utility class for training a Wasserstein GAN discriminator """
  def __init__(self, discriminator, input_size):
    self.disc = discriminator
    self.optim = torch.optim.Adam(self.disc.parameters(), 0.0003, (0.9, 0.999))
    summary(self.disc, input_size=input_size)
  def train_step(self, data_r, wts_r, data_g, wts_g):
    """ perform 1 training step of WGAN training
      data_r : (batch, ...) - ground truth samples from the world
      wts_r  : (batch,)     - importance weights for data_r, should sum to 1
      data_g : (batch, ...) - fake samples from the generator
      wts_g  : (batch,)     - importance weights for data_g, should sum to 1 """
    # enforce weights normalization
    assert abs(wts_r.detach().sum() - 1) < 1e-6
    assert abs(wts_g.detach().sum() - 1) < 1e-6
    # run 1 training step
    self.optim.zero_grad()
    y_r = self.disc(data_r)
    y_g = self.disc(data_g)
    loss = ((y_g*wts_g).sum() - (y_r*wts_r).sum()
      + λ*self.gradient_penalty(data_r, wts_r, data_g, wts_g))
    loss.backward()
    self.optim.step()
    return float(loss.detach())
  def gradient_penalty(self, data_r, wts_r, data_g, wts_g):
    """ The gradient penalty to enforce the Lipschitz condition. """
    data_mix = self.rand_weighted_mix(data_r, wts_r, data_g, wts_g)
    data_mix.requires_grad = True
    y = self.disc(data_mix)
    gradient = torch_grad(inputs=data_mix, outputs=y,
      grad_outputs=torch.ones(*y.shape),
      create_graph=True)[0]
    penalty = torch.relu(torch_vnorm(gradient) - 1.)**2
    return penalty
  def rand_weighted_mix(self, data_r, wts_r, data_g, wts_g, N=None):
    """ Create a batch of mixed data. The batchsize is N.
      Each item is a mix of two items from data_r and data_g,
      sampled according to probabilities from wts_r and wts_g.
      The amount of each item in the mix is uniformly random. """
    if N is None:
      N = wts_r.shape[0] + wts_g.shape[0]
    i_r = self.sample_indices_weighted(wts_r, N)
    i_g = self.sample_indices_weighted(wts_g, N)
    i_g = i_g[torch.randperm(N)] # shuffle
    epsilon = torch.rand(N, 1, 1) # different amount of mixing for each element
    return epsilon*data_r[i_r] + (1 - epsilon)*data_g[i_g]
  def sample_indices_weighted(self, wts, n):
    """ Smooth sample of indices according to the probabilities in wts.
      Samples n indices, each index is in [0...len(wts)-1].
      The overall batch of samples is smooth in that indices with a probability of 1/n
      are guaranteed to be sampled at least once, indices with a probability of 2/n at
      least twice, and so on. """
    sum_wts = wts.cumsum(dim=0)
    shifts = (torch.rand(1) + torch.arange(n).reshape(n, 1))/(n + 1)
    return (shifts > sum_wts).sum(1)


class WGAN:
  def __init__(self, generator, discriminator, latents_size, img_size):
    self.gen = generator
    self.optim = torch.optim.Adam(self.gen.parameters(), 0.0003, (0.9, 0.999))
    self.latents_size = latents_size
    summary(self.gen, latents_size) # hardcoded lat
    self.wgan_disc = WGANDisc(discriminator, img_size)
  def train_step(self, data_r, wts_r):
    loss_d = self.train_step_disc(data_r, wts_r)
    loss_g = self.train_step_gen()
    return "%f\t%f" % (loss_d, loss_g)
  def train_step_gen(self):
    self.optim.zero_grad()
    loss = -self.wgan_disc.disc(self.gen(self.get_latents())).mean()
    loss.backward()
    self.optim.step()
    return float(loss.detach())
  def train_step_disc(self, data_r, wts_r):
    data_g = self.gen(self.get_latents()).detach()
    batch = data_g.shape[0]
    wts_g = torch.ones(batch) / batch
    return self.wgan_disc.train_step(data_r, wts_r, data_g, wts_g)
  def get_latents(self):
    return torch.randn(*self.latents_size)


wgan = WGAN(Generator(), Discriminator(), (16, 128), (16, 28, 28))

for i, (img, _) in enumerate(mnist):
  loss = wgan.train_step(img, torch.tensor([1.]))
  print(i, "\t", loss)
  if i % 20 == 0:
    with torch.no_grad():
      img_gen = wgan.gen(wgan.get_latents())
      show_img(img_gen[0])


