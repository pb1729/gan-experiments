import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchinfo import summary
from torch.autograd import grad as torch_grad
from torch.linalg import vector_norm as torch_vnorm

from dataset import mnist
from tensorboard_viz import TensorBoard


device = "cuda"
batch = 128 # batch size
lr_d = 0.0001  # learning rate for discriminator
lr_g = 0.0003  # learning rate for generator
beta_1 = 0.5   # Adam parameter
beta_2 = 0.99  # Adam parameter
d_step_n = 1   # number of discriminator steps per generator step



class ResLayer(nn.Module):
  def __init__(self, n):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(n, n),
      nn.BatchNorm1d(n),
      nn.LeakyReLU(),
      nn.Linear(n, n, bias=False),
      nn.Dropout(p=0.3),
    )
  def forward(self, x):
    return x + self.layers(x)

class ConvResLayer(nn.Module):
  def __init__(self, n, h, w, kernsz):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Conv2d(n, n, kernsz, padding="same"),
      nn.BatchNorm2d(n),
      nn.LeakyReLU(),
      nn.Conv2d(n, n, kernsz, padding="same", bias=False),
      nn.Dropout(p=0.2),
    )
  def forward(self, x):
    return x + self.layers(x)

class Discriminator(nn.Module):
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Conv2d( 1, 32, 5, bias=False),
      ConvResLayer(32, 24, 24, 5),
      ConvResLayer(32, 24, 24, 5),
      ConvResLayer(32, 24, 24, 5),
      nn.AvgPool2d(4),
      nn.Flatten(),
      nn.Linear(1152, 400, bias=False),
      ResLayer(400),
      ResLayer(400),
      ResLayer(400),
      nn.Linear(400, 1, bias=False),
      nn.Sigmoid(),
    )
  def forward(self, x):
    return self.layers(x.reshape(-1, 1, 28, 28))


class Generator(nn.Module):
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(128, 400),
      ResLayer(400),
      ResLayer(400),
      ResLayer(400),
      nn.Linear(400, 32*7*7, bias=False),
      nn.Unflatten(1, [32, 7, 7]),
      nn.ConvTranspose2d(32, 32, 4, 4, bias=False),
      ConvResLayer(32, 28, 28, 5),
      ConvResLayer(32, 28, 28, 5),
      ConvResLayer(32, 28, 28, 5),
      nn.Conv2d(32, 1, 5, padding="same", bias=False),
      nn.Tanh()
    )
  def forward(self, z):
    return self.layers(z).reshape(-1, 28, 28)


class GANDisc:
  """ utility class for training a GAN discriminator """
  def __init__(self, discriminator, input_size):
    self.disc = discriminator
    #self.optim = torch.optim.Adam(self.disc.parameters(), lr_d, (beta_1, beta_2))
    self.optim = torch.optim.SGD(self.disc.parameters(), lr_d)
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
    # run 1 training step on the real data
    self.optim.zero_grad()
    y_r = self.disc(data_r)
    loss_r = -torch.log(y_r).mean()
    loss_r.backward()
    self.optim.step()
    # run 1 training step on the fake data
    self.optim.zero_grad()
    y_g = self.disc(data_g)
    loss_g = -torch.log(1. - y_g).mean()
    loss_g.backward()
    self.optim.step()
    return loss_r + loss_g
  def save(self, path):
    torch.save(self.disc, path + ".disc.pt")


class GAN:
  def __init__(self, generator, discriminator, latents_size, img_size):
    self.gen = generator
    self.optim = torch.optim.Adam(self.gen.parameters(), lr_g, (beta_1, beta_2))
    self.latents_size = latents_size
    summary(self.gen, latents_size) # hardcoded lat
    self.gan_disc = GANDisc(discriminator, img_size)
    self.cycle_counter = 0
  def transform(self, data_r, inst_str=0.8):
    """ center around 0 """
    return 2*data_r - 1.
  def train_step(self, data_r, wts_r):
    data_r = self.transform(data_r)
    self.cycle_counter = (self.cycle_counter - 1) % d_step_n
    loss_d = self.train_step_disc(data_r, wts_r)
    if self.cycle_counter == 0:
      loss_g = self.train_step_gen()
    else: loss_g = 0.
    return loss_d, loss_g
  def train_step_gen(self):
    self.optim.zero_grad()
    img_gen = self.gen(self.get_latents())
    y_g = self.gan_disc.disc(img_gen)
    loss = -torch.log(y_g).mean()
    loss.backward()
    self.optim.step()
    return loss.item()
  def train_step_disc(self, data_r, wts_r):
    data_g = self.gen(self.get_latents()).detach()
    batch = data_g.shape[0]
    wts_g = torch.ones(batch, device=device) / batch
    return self.gan_disc.train_step(data_r, wts_r, data_g, wts_g)
  def get_latents(self):
    return torch.randn(*self.latents_size, device=device)
  def save(self, path):
    torch.save(self.gen, path + ".gen.pt")
    self.gan_disc.save(path)
  @classmethod
  def load(cls, path, latents_size, img_size):
    return cls(torch.load(path + ".gen.pt").to(device), torch.load(path + ".disc.pt").to(device),
      latents_size, img_size)


def batchify(generator, batchsz, epochs=10):
  stack = []
  for epoch in range(epochs):
    for (img, _) in generator:
      if len(stack) >= batchsz:
        yield torch.cat(stack, dim=0)
        stack = []
      stack.append(img)


def main(save_path, load_path=None):
  if load_path is None:
    gan = GAN(Generator().to(device), Discriminator().to(device), (batch, 128), (batch, 28, 28))
  else:
    print("loading from path load_path...")
    gan = GAN.load(load_path, (batch, 128), (batch, 28, 28))
  
  board = TensorBoard()

  for i, img in enumerate(batchify(mnist, batch)):
    img = img.to(device)
    loss_d, loss_g = gan.train_step(img, torch.tensor([1.], device=device))
    print(f"{i}\t {loss_d:7.4f}\t {loss_g:7.4f}")
    board.scalar("loss_d", i, loss_d)
    board.scalar("loss_g", i, loss_g)
    if i % 100 == 0:
      print("saving...")
      gan.save(save_path)
      print("saved.")
      with torch.no_grad():
        img_raw = gan.transform(img)
        img_gen = gan.gen(gan.get_latents())
        board.img_grid("generated images %d" % i, torch.cat([img_raw, img_gen], dim=0)[:, None, :, :])



if __name__ == "__main__":
  from sys import argv
  main(*argv[1:])


