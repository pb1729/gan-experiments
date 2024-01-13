import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchinfo import summary
from torch.autograd import grad as torch_grad
from torch.linalg import vector_norm as torch_vnorm

from dataset import mnist
from tensorboard_viz import TensorBoard, DummyTensorBoard


device = "cuda"
batch = 128     # batch size
lr_d = 0.0002   # learning rate for discriminator
lr_g = 0.0002   # learning rate for generator
beta_1 = 0.5    # Adam parameter
beta_2 = 0.99   # Adam parameter

image_size = 28 # Spatial size of training images.
nc = 1          # Number of channels in the training images. For color images this is 3
nz = 100        # Size of z latent vector (i.e. size of generator input)
ngf = 64        # Size of feature maps in generator
ndf = 64        # Size of feature maps in discriminator


class Discriminator(nn.Module):
  def __init__(self):
    super(Discriminator, self).__init__()
    self.main = nn.Sequential(
      # input is ``(nc) x 28 x 28``
      nn.Conv2d(nc, ndf*2, 13, 1, bias=False),
      nn.LeakyReLU(0.2, inplace=True),
      # state size. ``(ndf*2) x 16 x 16``
      nn.Conv2d(ndf * 2, ndf * 4, 4, 2, padding=1, bias=False),
      #nn.LayerNorm([ndf * 4, 8, 8]),
      nn.BatchNorm2d(ndf*4),
      nn.LeakyReLU(0.2, inplace=True),
      # state size. ``(ndf*4) x 8 x 8``
      nn.Conv2d(ndf * 4, ndf * 8, 4, 2, padding=1, bias=False),
      #nn.LayerNorm([ndf * 8, 4, 4]),
      nn.BatchNorm2d(ndf*8),
      nn.LeakyReLU(0.2, inplace=True),
      # state size. ``(ndf*8) x 4 x 4``
      nn.Conv2d(ndf * 8, 1, 4, 1, bias=False),
      nn.Sigmoid()
    )
  def forward(self, input):
    return self.main(input.reshape(batch, nc, image_size, image_size)).reshape(batch, 1)

class Generator(nn.Module):
  def __init__(self):
    super(Generator, self).__init__()
    self.main = nn.Sequential(
      # input is Z, going into a convolution
      nn.ConvTranspose2d( nz, ngf * 8, 4, 1, bias=False),
      nn.BatchNorm2d(ngf*8),
      nn.ReLU(True),
      # state size. ``(ngf*8) x 4 x 4``
      nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, padding=1, bias=False),
      nn.BatchNorm2d(ngf*4),
      nn.ReLU(True),
      # state size. ``(ngf*4) x 8 x 8``
      nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, padding=1, bias=False),
      nn.BatchNorm2d(ngf*2),
      nn.ReLU(True),
      # state size. ``(ngf*2) x 16 x 16``
      nn.ConvTranspose2d( ngf * 2, nc, 13, 1, bias=False),
      # state size. ``(ngf) x 28 x 28``
      nn.Tanh()
    )
  def forward(self, input):
    return self.main(input.reshape(batch, nz, 1, 1)).reshape(batch, nc, image_size, image_size)

# custom weights initialization called on ``netG`` and ``netD``
def weights_init(m):
  classname = m.__class__.__name__
  if classname.find('Conv') != -1:
    nn.init.normal_(m.weight.data, 0.0, 0.02)
  elif classname.find('BatchNorm') != -1:
    nn.init.normal_(m.weight.data, 1.0, 0.02)
    nn.init.constant_(m.bias.data, 0)


class GANTrainer:
  def __init__(self, disc, gen):
    self.disc = disc
    self.gen  = gen
    self.init_optim()
  @staticmethod
  def load(path):
    states = torch.load(path)
    disc, gen = Discriminator().to(device), Generator().to(device)
    disc.load_state_dict(states["disc"])
    gen.load_state_dict(states["gen"])
    return GANTrainer(disc, gen)
  @staticmethod
  def makenew():
    disc, gen = Discriminator().to(device), Generator().to(device)
    disc.apply(weights_init)
    gen.apply(weights_init)
    return GANTrainer(disc, gen)
  def init_optim(self):
    self.optim_d = torch.optim.Adam(self.disc.parameters(), lr_d, (beta_1, beta_2))
    self.optim_g = torch.optim.Adam(self.gen.parameters(),  lr_g, (beta_1, beta_2))
  def save(self, path):
    torch.save({
        "disc": self.disc.state_dict(),
        "gen": self.gen.state_dict(),
      }, path)
  def get_latents(self, batchsz=batch):
    return torch.randn(batchsz, nz, device=device)
  def train_step(self, data):
    loss_d_r = self.disc_r_step(data)
    loss_d_g = self.disc_g_step()
    loss_g   = self.gen_step()
    return loss_d_r, loss_d_g, loss_g
  def disc_r_step(self, data):
    self.optim_d.zero_grad()
    p_real = self.disc(data)
    loss = -torch.log(p_real).mean()
    loss.backward()
    self.optim_d.step()
    return loss.item()
  def disc_g_step(self):
    self.optim_d.zero_grad()
    img_gen = self.gen(self.get_latents())
    p_real = self.disc(img_gen)
    loss = -torch.log(1. - p_real).mean()
    loss.backward()
    self.optim_d.step()
    return loss.item()
  def gen_step(self):
    self.optim_g.zero_grad()
    img_gen = self.gen(self.get_latents())
    p_real = self.disc(img_gen)
    loss = -torch.log(p_real).mean()
    loss.backward()
    self.optim_g.step()
    return loss.item()



# generators for prepping the training set:

def transform(mnist_batches):
  for imgs in mnist_batches:
    imgs_dev = imgs.to(device)[:, None, :, :] # move to gpu and reshape to have channels dim
    yield 2*imgs_dev - 1. # center data around 0

def batchify(generator, batchsz, epochs=50):
  stack = []
  for epoch in range(epochs):
    for (img, _) in generator:
      if len(stack) >= batchsz:
        yield torch.cat(stack, dim=0)
        stack = []
      stack.append(img)


def train(gan, save_path, board=None):
  """ train a GAN. inputs:
    gan       - a GANTrainer to be fed training data
    save_path - string, location where the model should be saved to
    board     - None, or a TensorBoard to record training progress """
  if board is None:
    board = DummyTensorBoard()
  for i, imgs in enumerate(transform(batchify(mnist, batch))):
    loss_d_r, loss_d_g, loss_g = gan.train_step(imgs)
    print(f"{i}\t ℒᴰ = {loss_d_r:05.4f} + {loss_d_g:05.4f}\t ℒᴳ = {loss_g:05.4f}")
    board.scalar("loss_d_r", i, loss_d_r)
    board.scalar("loss_d_g", i, loss_d_g)
    board.scalar("loss_d", i, loss_d_r + loss_d_g)
    board.scalar("loss_g", i, loss_g)
    if i % 100 == 0:
      print("saving...")
      gan.save(save_path)
      print("saved.")
      with torch.no_grad():
        img_gen = gan.gen(gan.get_latents())
        board.img_grid("generated images %d" % i,
          torch.cat([imgs, img_gen], dim=0))


def main(save_path, load_path=None):
  if load_path is None:
    gan = GANTrainer.makenew()
  else:
    gan = GANTrainer.load(load_path)
  board = TensorBoard()
  train(gan, save_path, board=board)


if __name__ == "__main__":
  from sys import argv
  main(*argv[1:])




