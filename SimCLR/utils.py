import os
from PIL import Image
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, ImageNet
from operations import Genotype, OPERATIONS
from PIL import ImageFilter
import random

class CIFAR10Pair(CIFAR10):
    """CIFAR10 Dataset.
    """

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos_1, pos_2, target

class CIFAR100Pair(CIFAR100):
    """CIFAR100 Dataset.
    """

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos_1, pos_2, target


CIFAR_train_transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

CIFAR_test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]

ImageNet_train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
    transforms.RandomApply([
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

ImageNet_test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])


def parse(weights, operation_set, parse_method):
  gene = []
  layers = weights.shape[0]
  for i in range(layers):
    if 'edge' in parse_method:
      topM = sorted(enumerate(weights[i]), key=lambda x: x[1])[-2:]
      gene.append([operation_set[topM[0][0]], operation_set[topM[1][0]]])
    elif 'sparse' in parse_method:
      best = sorted(enumerate(weights[i]), key=lambda x: x[1])[-1:][0]
      gene.append([operation_set[best[0]]])
    else:
      raise NotImplementedError("Unsupported parsing method: {}".format(parse_method))
  return gene


def parse_genotype(alphas, path = None, parse_method='sparse', model_name=''):
  primitives = list(OPERATIONS.keys())
  genes = parse(alphas, primitives, parse_method)
  genotype = Genotype(predictor=genes)

  if path is not None:
      if not os.path.exists(path):
          os.makedirs(path)
      print('Architecture parsing....\n', genotype)
      save_path = os.path.join(path, model_name + '_' + parse_method + '.txt')
      with open(save_path, "w+") as f:
          f.write(str(genotype))
          print('Save in :', save_path)
  return genotype