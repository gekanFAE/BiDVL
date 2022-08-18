# *-* coding:utf8 *-*
"""
Resnet based EBMs
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resblocks import DBlockOptimized, DBlock, SNLinear


class ResEBM32(nn.Module):
    def __init__(self, nf=128):
        super(ResEBM32, self).__init__()
        self.nf = nf

        # Build layers
        self.block1 = DBlockOptimized(3, self.nf)
        self.block2 = DBlock(self.nf, self.nf, downsample=True)
        self.block3 = DBlock(self.nf, self.nf, downsample=False)
        self.block4 = DBlock(self.nf, self.nf, downsample=False)
        self.l5 = SNLinear(self.nf, 1)
        self.activation = nn.ReLU(True)

        # Initialise the weights
        nn.init.xavier_uniform_(self.l5.weight.data, 1.0)

    def energy(self, v):
        e = self.forward(v)
        # e = F.softplus(e)
        return e

    def forward(self, x):
        out = x
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.activation(out)
        out = torch.sum(out, dim=(2, 3))
        out = self.l5(out)
        return out


class ResEBM64(nn.Module):
    def __init__(self, nf=1024):
        super(ResEBM64, self).__init__()
        self.nf = nf

        # Build layers
        self.block1 = DBlockOptimized(3, self.nf >> 4)
        self.block2 = DBlock(self.nf >> 4, self.nf >> 3, downsample=True)
        self.block3 = DBlock(self.nf >> 3, self.nf >> 2, downsample=True)
        self.block4 = DBlock(self.nf >> 2, self.nf >> 1, downsample=True)
        self.block5 = DBlock(self.nf >> 1, self.nf, downsample=True)
        self.l6 = SNLinear(self.nf, 1)
        self.activation = nn.ReLU(True)

        # Initialise the weights
        nn.init.xavier_uniform_(self.l6.weight.data, 1.0)

    def energy(self, v):
        e = self.forward(v)
        # e = F.softplus(e)
        return e

    def forward(self, x):
        out = x
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.activation(out)
        out = torch.sum(out, dim=(2, 3))
        output = self.l6(out)
        return output


class ResEBM128(nn.Module):
    def __init__(self, nf=1024):
        super(ResEBM128, self).__init__()
        self.nf = nf

        # Build layers
        self.block1 = DBlockOptimized(3, self.nf >> 4)
        self.block2 = DBlock(self.nf >> 4, self.nf >> 3, downsample=True)
        self.block3 = DBlock(self.nf >> 3, self.nf >> 2, downsample=True)
        self.block4 = DBlock(self.nf >> 2, self.nf >> 1, downsample=True)
        self.block5 = DBlock(self.nf >> 1, self.nf, downsample=True)
        self.block6 = DBlock(self.nf, self.nf, downsample=False)
        self.l7 = SNLinear(self.nf, 1)
        self.activation = nn.ReLU(True)

        # Initialise the weights
        nn.init.xavier_uniform_(self.l7.weight.data, 1.0)

    def energy(self, v):
        e = self.forward(v)
        # e = F.softplus(e)
        return e

    def forward(self, x):
        out = x
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.block6(out)
        out = self.activation(out)
        out = torch.sum(out, dim=(2, 3))
        output = self.l7(out)
        return output
