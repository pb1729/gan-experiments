# Tensorboard Visualization Code
#
#   --- Utilization: ---
#
# tensorboard --logdir=runs
# ssh -NfL 6006:localhost:6006 phillip@pico.phas.ubc.ca
# firefox http://localhost:6006

from torch.utils.tensorboard import SummaryWriter
import torchvision


LOG_DIR = "runs"


class DummyTensorBoard:
  """ dummy tensorboard that implements the same methods as a real one """
  def img_grid(self, label, images):
    pass
  def scalar(self, label, i, val):
    pass

class TensorBoard:
  """ logs various kinds of data to a tensor board """
  def __init__(self):
    self.writer = SummaryWriter(LOG_DIR)
    self.histories = {}
  def img_grid(self, label, images):
    grid = torchvision.utils.make_grid(images)
    self.writer.add_image(label, grid)
  def scalar(self, label, i, val):
    self.writer.add_scalar(label, val, i)


