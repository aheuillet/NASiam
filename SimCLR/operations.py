from collections import namedtuple
import torch
import torch.nn as nn

Genotype = namedtuple('Genotype', 'predictor')

OPERATIONS = {
    "maxpool_3x3": lambda in_dim, out_dim: op_wrapper(in_dim, out_dim, nn.Sequential(nn.MaxPool1d(3, stride=1, padding=1), nn.BatchNorm1d(out_dim))),
    "avgpool_3x3": lambda in_dim, out_dim: op_wrapper(in_dim, out_dim, nn.Sequential(nn.AvgPool1d(3, stride=1, padding=1), nn.BatchNorm1d(out_dim))),
    "identity": lambda in_dim, out_dim: op_wrapper(in_dim, out_dim, nn.Identity()),
    "linear_relu": lambda in_dim, out_dim: linear_wrapper(in_dim, out_dim, nn.ReLU(inplace=True)),
    #"linear_mish": lambda in_dim, out_dim: linear_wrapper(in_dim, out_dim, nn.Mish(inplace=True)),
    "linear_hardswish": lambda in_dim, out_dim: linear_wrapper(in_dim, out_dim, nn.Hardswish(inplace=True)),
  	"linear_silu": lambda in_dim, out_dim: linear_wrapper(in_dim, out_dim, nn.SiLU(inplace=True)),
  	"linear_elu": lambda in_dim, out_dim: linear_wrapper(in_dim, out_dim, nn.ELU(inplace=True))
}

def linear_wrapper(in_dim, out_dim, act):
  #act = nn.ReLU(inplace=True)
  if in_dim < out_dim:
      return nn.Linear(in_dim, out_dim)
  else:
      return nn.Sequential(nn.Linear(in_dim, out_dim, bias=False), nn.BatchNorm1d(out_dim), act)

def op_wrapper(in_dim, out_dim, op):
  act = nn.ReLU(inplace=True)
  if in_dim > out_dim:
    return nn.Sequential(nn.Linear(in_dim, out_dim, bias=False), nn.BatchNorm1d(out_dim), act, op)
  elif in_dim < out_dim:
    return nn.Sequential(nn.Linear(in_dim, out_dim), op) #remove batchnorm on predictor output to prevent instability
  else:
    return op

class MixedOp(nn.Module):
  def __init__(self, in_dim, out_dim):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    for primitive in OPERATIONS:
      op = OPERATIONS[primitive](in_dim, out_dim)
      self._ops.append(op)

  def forward(self, x, weights):
    return sum(w * op(x) for w, op in zip(weights, self._ops))

class Factor(nn.Module):
  def __init__(self, in_dim, init = 0.25):
    super(Factor, self).__init__()
    self.weight = nn.Parameter(torch.empty(in_dim, device="cuda").fill_(init))
  
  def forward(self, x):
    return self.weight*x

class Bias(nn.Module):
  def __init__(self, in_dim, init = 0.25):
    super(Factor, self).__init__()
    self.weight = nn.Parameter(torch.empty(in_dim, device="cuda").fill_(init))
  
  def forward(self, x):
    return self.weight + x