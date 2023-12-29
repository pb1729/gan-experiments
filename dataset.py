import torchvision.datasets as datasets
from torchvision.transforms import ToTensor

mnist = datasets.MNIST(root='./data', train=True, download=True,
    transform=ToTensor(),)


