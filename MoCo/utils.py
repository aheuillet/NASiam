import os
import numpy as np
import torch
#import pandas as pd
import random
#from tqdm import tqdm
import shutil
import torchvision.transforms as transforms
from torchvision.datasets import VisionDataset
from torch.autograd import Variable
from torchvision.transforms.transforms import Resize
#from auto_augment import CIFAR10Policy, ImageNetPolicy
from typing import Any, Callable, Optional
class AvgrageMeter(object):
  """
    Keeps track of most recent, average, sum, and count of a metric.
  """

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
  """Compute the top1 and top5 accuracy

  """
  maxk = max(topk)
  batch_size = target.size(0)

  # Return the k largest elements of the given input tensor
  # along a given dimension -> N * k
  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].reshape(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img

def _data_transforms_cifar(args):
  CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124] if args.dataset == 'cifar10' else [0.50707519, 0.48654887, 0.44091785]
  CIFAR_STD = [0.24703233, 0.24348505, 0.26158768] if args.dataset == 'cifar10' else [0.26733428, 0.25643846, 0.27615049]

  normalize_transform = [
      transforms.ToTensor(),
      transforms.Normalize(CIFAR_MEAN, CIFAR_STD)]
  
  if args.simsiam:
    random_transform = [
      transforms.RandomResizedCrop(32),
      transforms.RandomHorizontalFlip(),
      transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
      transforms.RandomGrayscale(p=0.2),
    ]
  else:
    random_transform = [
      transforms.RandomResizedCrop(32),
      transforms.RandomHorizontalFlip()
    ]

  # if args.auto_aug:
  #   random_transform += [CIFAR10Policy()]

  # if args.cutout:
  #   cutout_transform = [Cutout(args.cutout_length)]
  # else:
  #   cutout_transform = []

  train_transform = transforms.Compose(
      random_transform + normalize_transform #+ cutout_transform
  )

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
  return train_transform, valid_transform

def _data_transforms_imagenet(args):
  IMAGENET_MEAN = [0.485, 0.456, 0.406]
  IMAGENET_STD = [0.229, 0.224, 0.225]

  normalize_transform = [
      transforms.ToTensor(),
      transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]

  random_transform = [
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0.2),]

  # if args.auto_aug:
  #   random_transform += [ImageNetPolicy()]

  train_transform = transforms.Compose(
      random_transform + normalize_transform
  )

  valid_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

  return train_transform, valid_transform


def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for v in model.parameters())/1e6


def save_checkpoint(state, is_best, save):
  filename = os.path.join(save, 'checkpoint.pth.tar')
  torch.save(state, filename)
  if is_best:
    best_filename = os.path.join(save, 'model_best.pth.tar')
    shutil.copyfile(filename, best_filename)


def save(model, model_path):
  torch.save(model.state_dict(), model_path)


def load(model, model_path):
  model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob):
  if drop_prob > 0.:
    keep_prob = 1.-drop_prob
    mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
    x.div_(keep_prob)
    x.mul_(mask)
  return x


def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.makedirs(path)
  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    os.makedirs(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)

def calc_time(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    t, h = divmod(h, 24)
    return {'day': t, 'hour': h, 'minute': m, 'second': int(s)}

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


from moco.operations import Genotype, OPERATIONS
def parse_genotype(alphas, path = None, parse_method='sparse', model_name=''):
  primitives = list(OPERATIONS.keys())
  genes = parse(alphas, primitives, parse_method)
  genotype = Genotype(encoder=genes)

  if path is not None:
      if not os.path.exists(path):
          os.makedirs(path)
      print('Architecture parsing....\n', genotype)
      save_path = os.path.join(path, model_name + '_' + parse_method + '.txt')
      with open(save_path, "w+") as f:
          f.write(str(genotype))
          print('Save in :', save_path)
  return genotype
# if __name__ == "__main__":
#   parse_genotype(np.load('cell_arch_weights.npy'), 4, 4, path='genotypes/') 