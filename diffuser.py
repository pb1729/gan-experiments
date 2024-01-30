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
batch = 16      # batch size
lr = 0.0001     # learning rate


img_sz = 28     # Spatial size of training images.
nc = 1          # Number of channels in the training images. For color images this is 3
nf = 64         # Size of feature maps
shrink = 0.1    # Amount by which images are scaled down before adding noise
time_hdim = 48  # Number of frequencies in the time-pre-embedding
gen_steps = 40  # Number of steps taken to *generate* a new image

def sigma_sched(t):
  """ standard deviation of added noise as a function of timestep """
  return 1. - torch.cos(0.5*np.pi*t)
# epsilon value for variance normalization in the loss function:
sigma_epsilon = sigma_sched(torch.tensor(0.1)).item()


class ResLayer(nn.Module):
  """ see fig 1b in https://arxiv.org/pdf/1603.05027.pdf """
  def __init__(self, dim):
    super(ResLayer, self).__init__()
    self.dim = dim
    self.layers = nn.Sequential(
      nn.BatchNorm1d(dim),
      nn.LeakyReLU(negative_slope=0.1),
      nn.Dropout(p=0.3),
      nn.Linear(dim, dim),
      nn.BatchNorm1d(dim),
      nn.LeakyReLU(negative_slope=0.1),
      nn.Linear(dim, dim, bias=False),
    )
  def forward(self, x):
    return x + self.layers(x)


class ResLayer2d(nn.Module):
  """ see fig 1b in https://arxiv.org/pdf/1603.05027.pdf """
  def __init__(self, dim, kernel):
    super(ResLayer2d, self).__init__()
    self.dim = dim
    self.layers = nn.Sequential(
      nn.BatchNorm2d(dim),
      nn.LeakyReLU(negative_slope=0.1),
      nn.Dropout(p=0.1),
      nn.Conv2d(dim, dim, kernel, padding="same"),
      nn.BatchNorm2d(dim),
      nn.LeakyReLU(negative_slope=0.1),
      nn.Conv2d(dim, dim, kernel, padding="same", bias=False),
    )
  def forward(self, x):
    return x + self.layers(x)


class TimeEmbedding(nn.Module):
  def __init__(self, hdim, outdim):
    super(TimeEmbedding, self).__init__()
    self.hdim = hdim
    self.lin1 = nn.Linear(2*self.hdim, outdim)
  def raw_t_embed(self, t):
    """ t has shape (batch,) """
    ang_freqs = torch.exp(-torch.arange(self.hdim, device=device)/(self.hdim - 1))
    phases = t[:, None] * ang_freqs[None, :]
    return torch.cat([
      torch.sin(phases),
      torch.cos(phases),
    ], dim=1)
  def forward(self, t):
    """ t has shape (batch,) """
    return self.lin1(self.raw_t_embed(t))


class UNet(nn.Module):
  def __init__(self):
    super(UNet, self).__init__()
    self.embed_t_1 = TimeEmbedding(time_hdim, nf)
    self.inmap_1 = nn.Conv2d(nc, nf, 5)           # img_sz ==> img_sz - 4
    self.outmap_1 = nn.ConvTranspose2d(nf, nc, 5) # img_sz - 4 ==> img_sz
    self.proc_1_i = nn.Sequential(
      ResLayer2d(nf, 5),
      ResLayer2d(nf, 3),
      ResLayer2d(nf, 5))
    self.proc_1_o = nn.Sequential(
      ResLayer2d(nf, 5),
      ResLayer2d(nf, 3),
      ResLayer2d(nf, 5))
    self.embed_t_2 = TimeEmbedding(time_hdim, nf*2)
    self.inmap_2 = nn.Conv2d(nf, nf*2, 4, 2, padding=1)           # img_sz - 4 ==> (img_sz - 4)/2
    self.outmap_2 = nn.ConvTranspose2d(nf*2, nf, 4, 2, padding=1) # (img_sz - 4)/2 ==> img_sz - 4
    self.proc_2_i = nn.Sequential(
      ResLayer2d(nf*2, 5),
      ResLayer2d(nf*2, 3),
      ResLayer2d(nf*2, 5))
    self.proc_2_o = nn.Sequential(
      ResLayer2d(nf*2, 5),
      ResLayer2d(nf*2, 3),
      ResLayer2d(nf*2, 5))
    self.embed_t_3 = TimeEmbedding(time_hdim, nf*4)
    self.inmap_3 = nn.Conv2d(nf*2, nf*4, 4, 2, padding=1)           # (img_sz - 4)/2 ==> (img_sz - 4)/4
    self.outmap_3 = nn.ConvTranspose2d(nf*4, nf*2, 4, 2, padding=1) # (img_sz - 4)/4 ==> (img_sz - 4)/2
    self.proc_3_i = nn.Sequential(
      ResLayer2d(nf*4, 5),
      ResLayer2d(nf*4, 3),
      ResLayer2d(nf*4, 5))
    self.proc_3_o = nn.Sequential(
      ResLayer2d(nf*4, 5),
      ResLayer2d(nf*4, 3),
      ResLayer2d(nf*4, 5))
    self.embed_t_4 = TimeEmbedding(time_hdim, nf*8)
    self.inmap_4 = nn.Conv2d(nf*4, nf*8, 4, 2, padding=1)           # (img_sz - 4)/4 ==> (img_sz - 4)/8
    self.outmap_4 = nn.ConvTranspose2d(nf*8, nf*4, 4, 2, padding=1) # (img_sz - 4)/8 ==> (img_sz - 4)/4
    self.proc_4_i = nn.Sequential(
      ResLayer2d(nf*8, 3),
      ResLayer2d(nf*8, 3),
      ResLayer2d(nf*8, 3))
    self.proc_4_o = nn.Sequential(
      ResLayer2d(nf*8, 3),
      ResLayer2d(nf*8, 3),
      ResLayer2d(nf*8, 3))
    self.embed_t_5 = TimeEmbedding(time_hdim, nf*16)
    self.inmap_5 = nn.Sequential(
      nn.Flatten(),
      nn.Linear(nf*8*3*3, nf*16))
    self.outmap_5 = nn.Sequential(
      nn.Linear(nf*16, nf*8*3*3),
      nn.Unflatten(1, [nf*8, 3, 3]))
    self.proc_5_i = nn.Sequential(
      ResLayer(nf*16),
      ResLayer(nf*16),
      ResLayer(nf*16))
    self.proc_5_o = nn.Sequential(
      ResLayer(nf*16),
      ResLayer(nf*16),
      ResLayer(nf*16))
  def forward(self, x, t):
    y1 = self.proc_1_i(self.inmap_1(x) + self.embed_t_1(t)[:, :, None, None])
    y2 = self.proc_2_i(self.inmap_2(y1) + self.embed_t_2(t)[:, :, None, None])
    y3 = self.proc_3_i(self.inmap_3(y2) + self.embed_t_3(t)[:, :, None, None])
    y4 = self.proc_4_i(self.inmap_4(y3) + self.embed_t_4(t)[:, :, None, None])
    y5 = self.proc_5_i(self.inmap_5(y4) + self.embed_t_5(t))
    z5 = self.proc_5_o(y5                     + self.embed_t_5(t))
    z4 = self.proc_4_o(y4 + self.outmap_5(z5) + self.embed_t_4(t)[:, :, None, None])
    z3 = self.proc_3_o(y3 + self.outmap_4(z4) + self.embed_t_3(t)[:, :, None, None])
    z2 = self.proc_2_o(y2 + self.outmap_3(z3) + self.embed_t_2(t)[:, :, None, None])
    z1 = self.proc_1_o(y1 + self.outmap_2(z2) + self.embed_t_1(t)[:, :, None, None])
    return self.outmap_1(z1)



# generators for prepping the training set:

def transform(mnist_batches):
  for imgs in mnist_batches:
    imgs_dev = imgs.to(device)[:, None, :, :] # move to gpu and reshape to have channels dim
    yield 2*imgs_dev - 1. # center data around 0

def batchify(generator, batchsz, epochs=50):
  stack = []
  for epoch in range(epochs):
    print("epoch:", epoch)
    for (img, _) in generator:
      if len(stack) >= batchsz:
        yield torch.cat(stack, dim=0)
        stack = []
      stack.append(img)


# custom weights initialization
def weights_init(m):
  classname = m.__class__.__name__
  if classname.find('Conv') != -1:
    nn.init.normal_(m.weight.data, 0.0, 0.02)
  elif classname.find('BatchNorm') != -1:
    nn.init.normal_(m.weight.data, 1.0, 0.02)
    nn.init.constant_(m.bias.data, 0)


class DiffuserTrainer:
  def __init__(self, unet):
    self.unet = unet
    self.optim = torch.optim.Adam(self.unet.parameters(), lr)
  @staticmethod
  def load(path):
    states = torch.load(path)
    unet = UNet().to(device)
    unet.load_state_dict(states["unet"])
    return DiffuserTrainer(unet)
  @staticmethod
  def makenew():
    unet = UNet().to(device)
    unet.apply(weights_init)
    return DiffuserTrainer(unet)
  def save(self, path):
    torch.save({
        "unet": self.unet.state_dict(),
      }, path)
  def train_step(self, data):
    batch, *rest = data.shape
    assert tuple(rest) == (nc, img_sz, img_sz)
    # training step for UNet with squared error loss
    self.unet.zero_grad()
    t = torch.rand(batch, device=device)
    sigma = sigma_sched(t)[:, None, None, None]
    noise = sigma*torch.randn(batch, nc, img_sz, img_sz, device=device)
    noised_data = noise + shrink*data
    predicted_noise = self.unet(noised_data, t)
    rms_errs = torch.sqrt(((noise - predicted_noise)**2).mean(3).mean(2).mean(1))
    loss = (rms_errs/(sigma + sigma_epsilon)).mean()
    loss.backward()
    self.optim.step()
    return loss.item()
  def generate(self, bsz=4):
    with torch.no_grad():
      x = torch.zeros(bsz, nc, img_sz, img_sz, device=device)
      for i in range(gen_steps):
        x += torch.randn(bsz, nc, img_sz, img_sz, device=device)
        t = torch.ones(bsz, device=device)*(1. + i)/gen_steps
        x -= self.unet(x, t)
      return x/shrink


def train(trainer, save_path, board=None):
  """ train a GAN. inputs:
    gan       - a GANTrainer to be fed training data
    save_path - string, location where the model should be saved to
    board     - None, or a TensorBoard to record training progress """
  if board is None:
    board = DummyTensorBoard()
  for i, imgs in enumerate(transform(batchify(mnist, batch))):
    loss = trainer.train_step(imgs)
    print(f"{i}\t ℒ = {loss:05.4f}")
    board.scalar("loss", i, loss)
    if i % 100 == 0:
      print("saving...")
      trainer.save(save_path)
      print("saved.")
      with torch.no_grad():
        img_gen = trainer.generate(16)
        board.img_grid("real vs generated images %d" % i,
          torch.cat([imgs, img_gen], dim=0))


def main(save_path, load_path=None):
  if load_path is None:
    trainer = DiffuserTrainer.makenew()
  else:
    trainer = DiffuserTrainer.load(load_path)
  board = TensorBoard()
  train(trainer, save_path, board=board)


if __name__ == "__main__":
  from sys import argv
  main(*argv[1:])



