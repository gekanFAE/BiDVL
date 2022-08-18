# *-* coding:utf8 *-*
"""
Resnet based variational models
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import gauss_log_likelihood
from models.resblocks import DBlockOptimized, DBlock, GBlock


class ResEncoder32(nn.Module):
    def __init__(self, nz=128, nef=128):
        super(ResEncoder32, self).__init__()
        self.nz = nz
        self.nef = nef

        # Build encoder
        self.block1 = DBlockOptimized(3, self.nef)
        self.block2 = DBlock(self.nef, self.nef, downsample=True)
        self.block3 = DBlock(self.nef, self.nef, downsample=False)
        self.block4 = DBlock(self.nef, self.nef, downsample=False)
        self.activation = nn.ReLU(True)
        self.mu_layer = nn.Linear(self.nef, self.nz)
        self.var_layer = nn.Linear(self.nef, self.nz)
        self.mu_layer = nn.utils.spectral_norm(self.mu_layer)
        self.var_layer = nn.utils.spectral_norm(self.var_layer)

        # # Initialise the weights
        nn.init.xavier_uniform_(self.mu_layer.weight.data, 1.0)
        nn.init.xavier_uniform_(self.var_layer.weight.data, 1.0)

    def inference(self, v):
        mu, log_var = self.forward(v)
        return mu, log_var

    def log_q(self, h, mu, log_var):
        # compute log q(h|v)
        log_q = gauss_log_likelihood(h, mu, log_var)
        return log_q

    def forward(self, x):
        out = x
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.activation(out)
        out = torch.sum(out, dim=(2, 3))
        mu = self.mu_layer(out)
        log_var = self.var_layer(out)
        log_var = -F.softplus(log_var)
        return mu, log_var


class ResDecoder32(nn.Module):
    def __init__(self, nz=128, ndf=256, bottom_width=4):
        super(ResDecoder32, self).__init__()
        self.nz = nz
        self.ndf = ndf
        self.bottom_width = bottom_width

        # Build decoder
        self.l1 = nn.Linear(self.nz, (self.bottom_width ** 2) * self.ndf)
        self.block2 = GBlock(self.ndf, self.ndf, upsample=True)
        self.block3 = GBlock(self.ndf, self.ndf, upsample=True)
        self.block4 = GBlock(self.ndf, self.ndf, upsample=True)
        self.b5 = nn.BatchNorm2d(self.ndf)
        self.c5 = nn.Conv2d(self.ndf, 3, 3, 1, padding=1)
        self.activation = nn.ReLU(True)

        # Initialise the weights
        nn.init.xavier_uniform_(self.l1.weight.data, 1.0)
        nn.init.xavier_uniform_(self.c5.weight.data, 1.0)

    def sample(self, h):
        v = self.forward(h)
        return v

    def forward(self, x):
        out = x
        out = self.l1(out)
        out = out.view(x.shape[0], -1, self.bottom_width, self.bottom_width)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.b5(out)
        out = self.activation(out)
        out = torch.tanh(self.c5(out))
        return out


class ResEncoder128(nn.Module):
    def __init__(self, nz=128, nef=1024):
        super(ResEncoder128, self).__init__()
        self.nz = nz
        self.nef = nef

        # Build encoder
        self.block1 = DBlockOptimized(3, self.nef >> 4)
        self.block2 = DBlock(self.nef >> 4, self.nef >> 3, downsample=True)
        self.block3 = DBlock(self.nef >> 3, self.nef >> 2, downsample=True)
        self.block4 = DBlock(self.nef >> 2, self.nef >> 1, downsample=True)
        self.block5 = DBlock(self.nef >> 1, self.nef, downsample=True)
        self.block6 = DBlock(self.nef, self.nef, downsample=False)
        self.activation = nn.ReLU(True)
        self.mu_layer = nn.Linear(self.nef, self.nz, bias=False)
        self.var_layer = nn.Linear(self.nef, self.nz, bias=False)
        self.mu_layer = nn.utils.spectral_norm(self.mu_layer)
        self.var_layer = nn.utils.spectral_norm(self.var_layer)

        # Initialise the weights
        nn.init.xavier_uniform_(self.mu_layer.weight.data, 1.0)
        nn.init.xavier_uniform_(self.var_layer.weight.data, 1.0)

    def inference(self, v):
        mu, log_var = self.forward(v)
        return mu, log_var

    def log_q(self, h, mu, log_var):
        # compute log q(h|v)
        log_q = gauss_log_likelihood(h, mu, log_var)
        return log_q

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
        mu = self.mu_layer(out)
        log_var = self.var_layer(out)
        log_var = -F.softplus(log_var)
        return mu, log_var


class ResDecoder128(nn.Module):
    def __init__(self, nz=128, ndf=1024, bottom_width=4):
        super(ResDecoder128, self).__init__()
        self.nz = nz
        self.ndf = ndf
        self.bottom_width = bottom_width

        # Build decoder
        self.l1 = nn.Linear(self.nz, (self.bottom_width ** 2) * self.ndf)
        self.block2 = GBlock(self.ndf, self.ndf, upsample=True)
        self.block3 = GBlock(self.ndf, self.ndf >> 1, upsample=True)
        self.block4 = GBlock(self.ndf >> 1, self.ndf >> 2, upsample=True)
        self.block5 = GBlock(self.ndf >> 2, self.ndf >> 3, upsample=True)
        self.block6 = GBlock(self.ndf >> 3, self.ndf >> 4, upsample=True)
        self.b7 = nn.BatchNorm2d(self.ndf >> 4)
        self.c7 = nn.Conv2d(self.ndf >> 4, 3, 3, 1, padding=1)
        self.activation = nn.ReLU(True)

        # Initialise the weights
        nn.init.xavier_uniform_(self.l1.weight.data, 1.0)
        nn.init.xavier_uniform_(self.c7.weight.data, 1.0)

    def sample(self, h):
        v = self.forward(h)
        return v

    def forward(self, x):
        out = x
        out = self.l1(out)
        out = out.view(x.shape[0], -1, self.bottom_width, self.bottom_width)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.block6(out)
        out = self.b7(out)
        out = self.activation(out)
        out = torch.tanh(self.c7(out))
        return out
