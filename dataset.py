import torchvision.datasets as datasets
from torchvision.transforms import ToTensor

ROOT = "./data"

mnist = datasets.MNIST(root=ROOT, train=True, download=True,
    transform=ToTensor(),)

cifar10 = datasets.CIFAR10(root=ROOT, train=True, download=True,
    transform=ToTensor(),)

