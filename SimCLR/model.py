import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50
from operations import OPERATIONS, MixedOp

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

    def forward(self, x):
        weights = torch.softmax(self.alphas, dim=1)

        for i in range(self.num_layers):
            x = self._ops[i](x, weights[i])
        
        return x

class SearchModel(nn.Module):
    def __init__(self, feature_dim=128):
        super(SearchModel, self).__init__()
        self.f = []
        for name, module in resnet50().named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        self.cell_projector = SearchCell(2048, hidden_dim=512, out_dim=feature_dim, num_layers=6)
        # projection head
        self.g = nn.Sequential(self.cell_projector)
    
    def arch_parameters(self):
        return [self.cell_projector.alphas]

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)


class Model(nn.Module):
    def __init__(self, genotype, dataset="cifar10", feature_dim=128):
        super(Model, self).__init__()

        self.f = []
        if "cifar" in dataset:
            for name, module in resnet50().named_children():
                if name == 'conv1':
                    module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                    self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        self.cell_projector = Cell(2048, genotype.predictor, hidden_dim=512, out_dim=feature_dim)
        # projection head
        self.g = nn.Sequential(self.cell_projector)

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)
