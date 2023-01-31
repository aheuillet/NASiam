# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from simsiam.operations import OPERATIONS, MixedOp
from collections import OrderedDict

class SplitBatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, num_splits, **kw):
        super().__init__(num_features, **kw)
        self.num_splits = num_splits
        
    def forward(self, input):
        N, C, H, W = input.shape
        if self.training or not self.track_running_stats:
            running_mean_split = self.running_mean.repeat(self.num_splits)
            running_var_split = self.running_var.repeat(self.num_splits)
            outcome = nn.functional.batch_norm(
                input.view(-1, C * self.num_splits, H, W), running_mean_split, running_var_split, 
                self.weight.repeat(self.num_splits), self.bias.repeat(self.num_splits),
                True, self.momentum, self.eps).view(N, C, H, W)
            self.running_mean.data.copy_(running_mean_split.view(self.num_splits, C).mean(dim=0))
            self.running_var.data.copy_(running_var_split.view(self.num_splits, C).mean(dim=0))
            return outcome
        else:
            return nn.functional.batch_norm(
                input, self.running_mean, self.running_var, 
                self.weight, self.bias, False, self.momentum, self.eps)

def convert_cifar(base_encoder, feature_dim=128, zero_init=True):

    net = base_encoder(num_classes=feature_dim, zero_init_residual=zero_init)

    cifar = []
    for name, module in net.named_children():
        if name == 'conv1':
            module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        if isinstance(module, nn.MaxPool2d):
            continue
        if isinstance(module, nn.Linear):
            cifar.append(('flatten', nn.Flatten(1)))
        cifar.append((name, module))

    return nn.Sequential(OrderedDict(cifar))

class SearchCell(nn.Module):

    def __init__(self, dim, num_layers=9, hidden_dim=None, out_dim=None):
        super(SearchCell, self).__init__()
        if hidden_dim == None:
            hidden_dim = dim
        if out_dim == None:
            out_dim = dim
        self._ops = nn.ModuleList()

        self.num_ops = len(OPERATIONS)

        self.num_layers = num_layers

        self.alphas = torch.zeros(self.num_layers, self.num_ops, requires_grad=True, device="cuda")

        for i in range(self.num_layers):
            if i == 0:
                op = MixedOp(dim, hidden_dim)
            elif i == self.num_layers-1:
                op = MixedOp(hidden_dim, out_dim)
            else:
                 op = MixedOp(hidden_dim, hidden_dim)
            self._ops.append(op)
    
    def alpha_regularization(self, norm="L1", weight_decay=1e-3):
        if norm not in ["L1", "L2"]:
            raise ValueError("Norm not supported")
        loss = torch.tensor(0., device="cuda")
        for edge in self.alphas:
            for i,o in enumerate(edge):
                if i < 3:
                    loss += abs(o) if norm == "L1" else o*o
        return loss*weight_decay

    def forward(self, x):
        weights = torch.softmax(self.alphas, dim=1)

        for i in range(self.num_layers):
            x = self._ops[i](x, weights[i])
        
        return x

class Cell(nn.Module):

    def __init__(self, dim, genes, hidden_dim=None, out_dim=None):
        super(Cell, self).__init__()
        if hidden_dim == None:
            hidden_dim = dim
        if out_dim == None:
            out_dim = dim
        self._ops = nn.ModuleList()

        self.num_layers = len(genes)
        for i in range(self.num_layers):
            ops = nn.ModuleList()
            if i == 0:
                dim1 = dim
                dim2 = hidden_dim
            elif i == self.num_layers-1:
                dim1 = hidden_dim
                dim2 = out_dim
            else:
                dim1 = hidden_dim
                dim2 = hidden_dim
            op = genes[i]
            ops.append(OPERATIONS[op[0]](dim1, dim2))
            self._ops.append(ops)

    def forward(self, x):
        for l in self._ops:
            x = sum([op(x) for op in l])
        return x

class SimSiam(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, base_encoder, genotype, dim=2048, pred_dim=512):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(SimSiam, self).__init__()

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        if "resnet18" in str(base_encoder):
            self.encoder = convert_cifar(base_encoder, dim)
        else:
            self.encoder = base_encoder(num_classes=dim, zero_init_residual=True)

        # build a 3-layer projector
        prev_dim = self.encoder.fc.weight.shape[1]

        self.cell_encoder = Cell(prev_dim, genotype.encoder)
        self.encoder.fc = nn.Sequential(self.cell_encoder,
                                        self.encoder.fc,
                                        nn.BatchNorm1d(dim, affine=False)) # output layer
        self.encoder.fc[1].bias.requires_grad = False # hack: not use bias as it is followed by BN

        self.cell_predictor = Cell(dim, genotype.predictor, hidden_dim=pred_dim)

        # build a 2-layer predictor
        self.predictor = nn.Sequential(self.cell_predictor) # output layer

    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """

        # compute features for one view
        z1 = self.encoder(x1) # NxC
        z2 = self.encoder(x2) # NxC

        p1 = self.predictor(z1) # NxC
        p2 = self.predictor(z2) # NxC

        return p1, p2, z1.detach(), z2.detach()
    
    def compute_features(self, x):
        q = self.encoder(x)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)  # already normalized
        return q


class SimSiamSearch(nn.Module):
    """
    Build a SimSiam search proxy model.
    """
    def __init__(self, base_encoder, dim=2048, pred_dim=512):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(SimSiamSearch, self).__init__()

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        if "resnet18" in str(base_encoder):
            self.encoder = convert_cifar(base_encoder, dim)
        else:
            self.encoder = base_encoder(num_classes=dim, zero_init_residual=True)

        prev_dim = self.encoder.fc.weight.shape[1]
        self.cell_encoder = SearchCell(prev_dim, num_layers=6) 
        self.encoder.fc = nn.Sequential(self.cell_encoder,
                                        self.encoder.fc,
                                        nn.BatchNorm1d(dim, affine=False)) # output layer
        self.encoder.fc[1].bias.requires_grad = False # hack: not use bias as it is followed by BN

        self.cell_predictor = SearchCell(dim, hidden_dim=pred_dim, num_layers=4)
        
        self.predictor = nn.Sequential(self.cell_predictor) # output layer
    
    def arch_parameters(self):
        return [self.cell_encoder.alphas, self.cell_predictor.alphas]
    
    def alpha_regularization(self, norm, weight_decay):
        return self.cell_encoder.alpha_regularization(norm, weight_decay) + self.cell_predictor.alpha_regularization(norm, weight_decay)
    
    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """

        # compute features for one view
        z1 = self.encoder(x1) # NxC
        z2 = self.encoder(x2) # NxC

        p1 = self.predictor(z1) # NxC
        p2 = self.predictor(z2) # NxC

        return p1, p2, z1.detach(), z2.detach()

class StandardSimSiam(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, base_encoder, dim=2048, pred_dim=512):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(StandardSimSiam, self).__init__()

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        if "resnet18" in str(base_encoder):
            self.encoder = convert_cifar(base_encoder, dim)
        else:
            self.encoder = base_encoder(num_classes=dim, zero_init_residual=True)

        # build a 3-layer projector
        prev_dim = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        self.encoder.fc,
                                        nn.BatchNorm1d(dim, affine=False)) # output layer
        self.encoder.fc[6].bias.requires_grad = False # hack: not use bias as it is followed by BN

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                        nn.BatchNorm1d(pred_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(pred_dim, dim)) # output layer

    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """

        # compute features for one view
        z1 = self.encoder(x1) # NxC
        z2 = self.encoder(x2) # NxC

        p1 = self.predictor(z1) # NxC
        p2 = self.predictor(z2) # NxC

        return p1, p2, z1.detach(), z2.detach()