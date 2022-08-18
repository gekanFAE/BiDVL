# *-* coding:utf8 *-*
"""
Energy based models
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvJointE(nn.Module):
    def __init__(self, v_shape, h_dim, basec, activate):
        from torch.nn.utils import spectral_norm
        super(ConvJointE, self).__init__()
        v_inc = v_shape[1]
        h_inc = h_dim
        self.vnet = nn.Sequential(
            spectral_norm(nn.Conv2d(v_inc, basec, 3, 1, 1, bias=True)),
            nn.LeakyReLU(0.1, inplace=True),
            spectral_norm(nn.Conv2d(basec, basec, 4, 2, 1, bias=True)),
            nn.LeakyReLU(0.1, inplace=True),
            spectral_norm(nn.Conv2d(basec, basec * 2, 3, 1, 1, bias=True)),
            nn.LeakyReLU(0.1, inplace=True),
            spectral_norm(nn.Conv2d(basec * 2, basec * 2, 4, 2, 1, bias=True)),
            nn.LeakyReLU(0.1, inplace=True),
            spectral_norm(nn.Conv2d(basec * 2, basec * 4, 3, 1, 1, bias=True)),
            nn.LeakyReLU(0.1, inplace=True),
            spectral_norm(nn.Conv2d(basec * 4, basec * 4, 4, 2, 1, bias=True)),
            nn.LeakyReLU(0.1, inplace=True),
            spectral_norm(nn.Conv2d(basec * 4, basec * 8, 3, 1, 1, bias=True)),
            nn.LeakyReLU(0.1, inplace=True),
            spectral_norm(nn.Conv2d(basec * 8, basec * 8, 4, 1, 0)),
        )
        self.hnet = nn.Sequential(
            spectral_norm(nn.Conv2d(h_inc, basec, 1, 1, 0)),
            nn.LeakyReLU(0.1, inplace=True),
            spectral_norm(nn.Conv2d(basec, basec, 1, 1, 0)),
            nn.LeakyReLU(0.1, inplace=True),
            spectral_norm(nn.Conv2d(basec, basec * 2, 1, 1, 0)),
            nn.LeakyReLU(0.1, inplace=True),
            spectral_norm(nn.Conv2d(basec * 2, basec * 2, 1, 1, 0)),
            nn.LeakyReLU(0.1, inplace=True),
            spectral_norm(nn.Conv2d(basec * 2, basec * 4, 1, 1, 0)),
            nn.LeakyReLU(0.1, inplace=True),
            spectral_norm(nn.Conv2d(basec * 4, basec * 4, 1, 1, 0)),
            nn.LeakyReLU(0.1, inplace=True),
            spectral_norm(nn.Conv2d(basec * 4, basec * 8, 1, 1, 0)),
            nn.LeakyReLU(0.1, inplace=True),
            spectral_norm(nn.Conv2d(basec * 8, basec * 8, 1, 1, 0)),
        )
        self.enet = nn.Sequential(
            spectral_norm(nn.Conv2d(basec * 16, basec * 16, 1, 1, 0)),
            nn.LeakyReLU(0.1, inplace=True),
            spectral_norm(nn.Conv2d(basec * 16, 1, 1, 1, 0)),
        )
        self.activate = activate

    def forward(self, v, h):
        v_out = self.vnet(v)
        h_out = self.hnet(h)
        vh_out = torch.cat([v_out, h_out], dim=1)
        e = self.enet(vh_out)
        return e

    def energy(self, v, h):
        e = self.forward(v, h)
        if self.activate == 'tanh':
            e = F.tanh(-e.squeeze())
        elif self.activate == 'sigmoid':
            e = F.sigmoid(e.squeeze())
        elif self.activate == 'identity':
            e = e.squeeze()
        elif self.activate == 'softplus':
            e = F.softplus(e.squeeze())
        return e


class ConvE(nn.Module):
    def __init__(self, v_shape, basec, activate):
        from torch.nn.utils import spectral_norm
        super(ConvE, self).__init__()
        v_inc = v_shape[1]
        if v_shape[2] == 32:
            self.vnet = nn.Sequential(
                spectral_norm(nn.Conv2d(v_inc, basec, 3, 1, 1, bias=True)),
                nn.LeakyReLU(0.1, inplace=True),
                spectral_norm(nn.Conv2d(basec, basec, 4, 2, 1, bias=True)),
                nn.LeakyReLU(0.1, inplace=True),
                spectral_norm(nn.Conv2d(basec, basec * 2, 3, 1, 1, bias=True)),
                nn.LeakyReLU(0.1, inplace=True),
                spectral_norm(nn.Conv2d(basec * 2, basec * 2, 4, 2, 1, bias=True)),
                nn.LeakyReLU(0.1, inplace=True),
                spectral_norm(nn.Conv2d(basec * 2, basec * 4, 3, 1, 1, bias=True)),
                nn.LeakyReLU(0.1, inplace=True),
                spectral_norm(nn.Conv2d(basec * 4, basec * 4, 4, 2, 1, bias=True)),
                nn.LeakyReLU(0.1, inplace=True),
                spectral_norm(nn.Conv2d(basec * 4, basec * 8, 3, 1, 1, bias=True)),
                nn.LeakyReLU(0.1, inplace=True),
                spectral_norm(nn.Conv2d(basec * 8, basec * 8, 4, 1, 0)),
            )
            self.enet = nn.Sequential(
                spectral_norm(nn.Conv2d(basec * 8, basec * 8, 1, 1, 0)),
                nn.LeakyReLU(0.1, inplace=True),
                spectral_norm(nn.Conv2d(basec * 8, 1, 1, 1, 0)),
            )
        elif v_shape[2] == 64:
            self.vnet = nn.Sequential(
                spectral_norm(nn.Conv2d(v_inc, basec, 3, 1, 1, bias=True)),
                nn.LeakyReLU(0.1, inplace=True),
                spectral_norm(nn.Conv2d(basec, basec, 4, 2, 1, bias=True)),
                nn.LeakyReLU(0.1, inplace=True),
                spectral_norm(nn.Conv2d(basec, basec * 2, 3, 1, 1, bias=True)),
                nn.LeakyReLU(0.1, inplace=True),
                spectral_norm(nn.Conv2d(basec * 2, basec * 2, 4, 2, 1, bias=True)),
                nn.LeakyReLU(0.1, inplace=True),
                spectral_norm(nn.Conv2d(basec * 2, basec * 4, 3, 1, 1, bias=True)),
                nn.LeakyReLU(0.1, inplace=True),
                spectral_norm(nn.Conv2d(basec * 4, basec * 4, 4, 2, 1, bias=True)),
                nn.LeakyReLU(0.1, inplace=True),
                spectral_norm(nn.Conv2d(basec * 4, basec * 8, 3, 1, 1, bias=True)),
                nn.LeakyReLU(0.1, inplace=True),
                spectral_norm(nn.Conv2d(basec * 8, basec * 8, 4, 2, 1, bias=True)),
                nn.LeakyReLU(0.1, inplace=True),
                spectral_norm(nn.Conv2d(basec * 8, basec * 16, 3, 1, 1, bias=True)),
                nn.LeakyReLU(0.1, inplace=True),
                spectral_norm(nn.Conv2d(basec * 16, basec * 16, 4, 1, 0)),
            )
            self.enet = nn.Sequential(
                spectral_norm(nn.Conv2d(basec * 16, basec * 16, 1, 1, 0)),
                nn.LeakyReLU(0.1, inplace=True),
                spectral_norm(nn.Conv2d(basec * 16, 1, 1, 1, 0)),
            )
        elif v_shape[2] == 128:
            self.vnet = nn.Sequential(
                spectral_norm(nn.Conv2d(v_inc, basec, 3, 1, 1, bias=True)),
                nn.LeakyReLU(0.1, inplace=True),
                spectral_norm(nn.Conv2d(basec, basec, 4, 2, 1, bias=True)),
                nn.LeakyReLU(0.1, inplace=True),
                spectral_norm(nn.Conv2d(basec, basec * 2, 3, 1, 1, bias=True)),
                nn.LeakyReLU(0.1, inplace=True),
                spectral_norm(nn.Conv2d(basec * 2, basec * 2, 4, 2, 1, bias=True)),
                nn.LeakyReLU(0.1, inplace=True),
                spectral_norm(nn.Conv2d(basec * 2, basec * 4, 3, 1, 1, bias=True)),
                nn.LeakyReLU(0.1, inplace=True),
                spectral_norm(nn.Conv2d(basec * 4, basec * 4, 4, 2, 1, bias=True)),
                nn.LeakyReLU(0.1, inplace=True),
                spectral_norm(nn.Conv2d(basec * 4, basec * 8, 3, 1, 1, bias=True)),
                nn.LeakyReLU(0.1, inplace=True),
                spectral_norm(nn.Conv2d(basec * 8, basec * 8, 4, 2, 1, bias=True)),
                nn.LeakyReLU(0.1, inplace=True),
                spectral_norm(nn.Conv2d(basec * 8, basec * 16, 3, 1, 1, bias=True)),
                nn.LeakyReLU(0.1, inplace=True),
                spectral_norm(nn.Conv2d(basec * 16, basec * 16, 4, 2, 1, bias=True)),
                nn.LeakyReLU(0.1, inplace=True),
                spectral_norm(nn.Conv2d(basec * 16, basec * 32, 3, 1, 1, bias=True)),
                nn.LeakyReLU(0.1, inplace=True),
                spectral_norm(nn.Conv2d(basec * 32, basec * 32, 4, 1, 0)),
            )
            self.enet = nn.Sequential(
                spectral_norm(nn.Conv2d(basec * 32, basec * 32, 1, 1, 0)),
                nn.LeakyReLU(0.1, inplace=True),
                spectral_norm(nn.Conv2d(basec * 32, 1, 1, 1, 0)),
            )
        else:
            raise NotImplementedError
        self.activate = activate

    def forward(self, v):
        v_out = self.vnet(v)
        e = self.enet(v_out)
        return e

    def energy(self, v):
        e = self.forward(v)
        if self.activate == 'tanh':
            e = F.tanh(-e.squeeze())
        elif self.activate == 'sigmoid':
            e = F.sigmoid(e.squeeze())
        elif self.activate == 'identity':
            e = e.squeeze()
        elif self.activate == 'softplus':
            e = F.softplus(e.squeeze())
        return e


class ConvDoubleE(nn.Module):
    def __init__(self, v_shape, h_dim, basec, activate):
        from torch.nn.utils import spectral_norm
        super(ConvDoubleE, self).__init__()
        v_inc = v_shape[1]
        h_inc = h_dim
        self.vnet = nn.Sequential(
            spectral_norm(nn.Conv2d(v_inc, basec, 3, 1, 1, bias=True)),
            nn.LeakyReLU(0.1, inplace=True),
            spectral_norm(nn.Conv2d(basec, basec, 4, 2, 1, bias=True)),
            nn.LeakyReLU(0.1, inplace=True),
            spectral_norm(nn.Conv2d(basec, basec * 2, 3, 1, 1, bias=True)),
            nn.LeakyReLU(0.1, inplace=True),
            spectral_norm(nn.Conv2d(basec * 2, basec * 2, 4, 2, 1, bias=True)),
            nn.LeakyReLU(0.1, inplace=True),
            spectral_norm(nn.Conv2d(basec * 2, basec * 4, 3, 1, 1, bias=True)),
            nn.LeakyReLU(0.1, inplace=True),
            spectral_norm(nn.Conv2d(basec * 4, basec * 4, 4, 2, 1, bias=True)),
            nn.LeakyReLU(0.1, inplace=True),
            spectral_norm(nn.Conv2d(basec * 4, basec * 8, 3, 1, 1, bias=True)),
            nn.LeakyReLU(0.1, inplace=True),
            spectral_norm(nn.Conv2d(basec * 8, basec * 8, 4, 1, 0, bias=True)),
            nn.LeakyReLU(0.1, inplace=True),
            spectral_norm(nn.Conv2d(basec * 8, basec * 8, 1, 1, 0)),
            nn.LeakyReLU(0.1, inplace=True),
            spectral_norm(nn.Conv2d(basec * 8, 1, 1, 1, 0)),
        )
        self.hnet = nn.Sequential(
            spectral_norm(nn.Conv2d(h_inc, basec, 1, 1, 0)),
            nn.LeakyReLU(0.1, inplace=True),
            spectral_norm(nn.Conv2d(basec, basec, 1, 1, 0)),
            nn.LeakyReLU(0.1, inplace=True),
            spectral_norm(nn.Conv2d(basec, basec * 2, 1, 1, 0)),
            nn.LeakyReLU(0.1, inplace=True),
            spectral_norm(nn.Conv2d(basec * 2, basec * 2, 1, 1, 0)),
            nn.LeakyReLU(0.1, inplace=True),
            spectral_norm(nn.Conv2d(basec * 2, basec * 4, 1, 1, 0)),
            nn.LeakyReLU(0.1, inplace=True),
            spectral_norm(nn.Conv2d(basec * 4, basec * 4, 1, 1, 0)),
            nn.LeakyReLU(0.1, inplace=True),
            spectral_norm(nn.Conv2d(basec * 4, basec * 8, 1, 1, 0)),
            nn.LeakyReLU(0.1, inplace=True),
            spectral_norm(nn.Conv2d(basec * 8, basec * 8, 1, 1, 0)),
            nn.LeakyReLU(0.1, inplace=True),
            spectral_norm(nn.Conv2d(basec * 8, 1, 1, 1, 0)),
        )
        self.activate = activate

    def forward(self, x, mode):
        if mode == 'v':
            e = self.vnet(x)
        elif mode == 'h':
            e = self.hnet(x)
        else:
            raise NotImplementedError
        return e

    def v_energy(self, x):
        e = self.forward(x, 'v')
        if self.activate == 'tanh':
            e = F.tanh(-e.squeeze())
        elif self.activate == 'sigmoid':
            e = F.sigmoid(e.squeeze())
        elif self.activate == 'identity':
            e = e.squeeze()
        elif self.activate == 'softplus':
            e = F.softplus(e.squeeze())
        return e

    def h_energy(self, x):
        e = self.forward(x, 'h')
        if self.activate == 'tanh':
            e = F.tanh(-e.squeeze())
        elif self.activate == 'sigmoid':
            e = F.sigmoid(e.squeeze())
        elif self.activate == 'identity':
            e = e.squeeze()
        elif self.activate == 'softplus':
            e = F.softplus(e.squeeze())
        return e
