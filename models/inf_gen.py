# *-* coding:utf8 *-*
"""
Variational models
"""

import torch
import torch.nn as nn
from utils import reparemetrize, gauss_log_likelihood


class GaussQ(nn.Module):
    def __init__(self, v_shape, h_dim):
        super(GaussQ, self).__init__()
        self.v_shape = v_shape
        self.h_dim = h_dim

    def moments(self, v):
        # get mu and log_std
        raise NotImplementedError

    def log_q(self, h, mu, log_var):
        # compute log q(h|v)
        log_q = gauss_log_likelihood(h, mu, log_var)
        return log_q

    def inference(self, v):
        # sample from q(h|v) by reparametric trick
        mu, log_std = self.moments(v)
        h = reparemetrize(mu, log_std)
        return h


class ConvGaussQ(GaussQ):
    def __init__(self, v_shape, h_dim, basec):
        super(ConvGaussQ, self).__init__(v_shape, h_dim)
        assert len(v_shape) == 4, 'v_shape should be [b, c, h, w]'
        inc = v_shape[1]
        outc = h_dim
        self.qnet = nn.Sequential(
            nn.Conv2d(inc, basec, 3, 1, 1, bias=False),
            nn.BatchNorm2d(basec),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(basec, basec * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(basec * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(basec * 2, basec * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(basec * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(basec * 4, basec * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(basec * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.mu_layer = nn.Conv2d(basec * 8, outc, 4, 1, 0)
        self.var_layer = nn.Conv2d(basec * 8, outc, 4, 1, 0)

    def forward(self, v):
        out = self.qnet(v)
        mu = self.mu_layer(out)
        log_var = self.var_layer(out)
        return mu, log_var

    def moments(self, v):
        return self.forward(v)


class ConvGenerP(nn.Module):
    def __init__(self, v_shape, h_dim, basec):
        super(ConvGenerP, self).__init__()
        assert len(v_shape) == 4, 'v_shape should be [b, c, h, w]'
        outc = v_shape[1]
        inc = h_dim
        self.pnet = nn.Sequential(
            nn.ConvTranspose2d(inc, basec * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(basec * 8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(basec * 8, basec * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(basec * 8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(basec * 8, basec * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(basec * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(basec * 4, basec * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(basec * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(basec * 2, outc, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, h):
        v = self.pnet(h)
        return v

    def sample(self, h):
        return self.forward(h)


class VAE(nn.Module):
    def __init__(self, v_shape, h_dim, Qbasec, Pbasec):
        super(VAE, self).__init__()
        assert len(v_shape) == 4, 'v_shape should be [b, c, h, w]'

        if v_shape[2] == 32:
            inc = v_shape[1]
            outc = h_dim
            self.qnet = nn.Sequential(
                nn.Conv2d(inc, Qbasec, 3, 1, 1, bias=False),
                nn.BatchNorm2d(Qbasec),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(Qbasec, Qbasec * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(Qbasec * 2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(Qbasec * 2, Qbasec * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(Qbasec * 4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(Qbasec * 4, Qbasec * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(Qbasec * 8),
                nn.LeakyReLU(0.2, inplace=True),
            )
            self.mu_layer = nn.Conv2d(Qbasec * 8, outc, 4, 1, 0)
            self.var_layer = nn.Conv2d(Qbasec * 8, outc, 4, 1, 0)

            inc = h_dim
            outc = v_shape[1]
            self.pnet = nn.Sequential(
                nn.ConvTranspose2d(inc, Pbasec * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(Pbasec * 8),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(Pbasec * 8, Pbasec * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(Pbasec * 8),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(Pbasec * 8, Pbasec * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(Pbasec * 4),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(Pbasec * 4, Pbasec * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(Pbasec * 2),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(Pbasec * 2, outc, 3, 1, 1),
                nn.Tanh()
            )
        elif v_shape[2] == 64:
            inc = v_shape[1]
            outc = h_dim
            self.qnet = nn.Sequential(
                nn.Conv2d(inc, Qbasec, 3, 1, 1, bias=False),
                nn.BatchNorm2d(Qbasec),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(Qbasec, Qbasec * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(Qbasec * 2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(Qbasec * 2, Qbasec * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(Qbasec * 4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(Qbasec * 4, Qbasec * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(Qbasec * 8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(Qbasec * 8, Qbasec * 16, 4, 2, 1, bias=False),
                nn.BatchNorm2d(Qbasec * 16),
                nn.LeakyReLU(0.2, inplace=True),
            )
            self.mu_layer = nn.Conv2d(Qbasec * 16, outc, 4, 1, 0)
            self.var_layer = nn.Conv2d(Qbasec * 16, outc, 4, 1, 0)

            inc = h_dim
            outc = v_shape[1]
            self.pnet = nn.Sequential(
                nn.ConvTranspose2d(inc, Pbasec * 16, 4, 1, 0, bias=False),
                nn.BatchNorm2d(Pbasec * 16),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(Pbasec * 16, Pbasec * 16, 4, 2, 1, bias=False),
                nn.BatchNorm2d(Pbasec * 16),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(Pbasec * 16, Pbasec * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(Pbasec * 8),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(Pbasec * 8, Pbasec * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(Pbasec * 4),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(Pbasec * 4, Pbasec * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(Pbasec * 2),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(Pbasec * 2, outc, 3, 1, 1),
                nn.Tanh()
            )
        elif v_shape[2] == 128:
            inc = v_shape[1]
            outc = h_dim
            self.qnet = nn.Sequential(
                nn.Conv2d(inc, Qbasec, 3, 1, 1, bias=False),
                nn.BatchNorm2d(Qbasec),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(Qbasec, Qbasec * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(Qbasec * 2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(Qbasec * 2, Qbasec * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(Qbasec * 4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(Qbasec * 4, Qbasec * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(Qbasec * 8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(Qbasec * 8, Qbasec * 16, 4, 2, 1, bias=False),
                nn.BatchNorm2d(Qbasec * 16),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(Qbasec * 16, Qbasec * 32, 4, 2, 1, bias=False),
                nn.BatchNorm2d(Qbasec * 32),
                nn.LeakyReLU(0.2, inplace=True),
            )
            self.mu_layer = nn.Conv2d(Qbasec * 32, outc, 4, 1, 0)
            self.var_layer = nn.Conv2d(Qbasec * 32, outc, 4, 1, 0)

            inc = h_dim
            outc = v_shape[1]
            self.pnet = nn.Sequential(
                nn.ConvTranspose2d(inc, Pbasec * 32, 4, 1, 0, bias=False),
                nn.BatchNorm2d(Pbasec * 32),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(Pbasec * 32, Pbasec * 32, 4, 2, 1, bias=False),
                nn.BatchNorm2d(Pbasec * 32),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(Pbasec * 32, Pbasec * 16, 4, 2, 1, bias=False),
                nn.BatchNorm2d(Pbasec * 16),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(Pbasec * 16, Pbasec * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(Pbasec * 8),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(Pbasec * 8, Pbasec * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(Pbasec * 4),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(Pbasec * 4, Pbasec * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(Pbasec * 2),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(Pbasec * 2, outc, 3, 1, 1),
                nn.Tanh()
            )
        else:
            raise NotImplementedError

    def inference(self, v):
        return self.forward(v, mode='inference')

    def sample(self, h):
        return self.forward(h, mode='sample')

    def log_q(self, h, mu, log_var):
        # compute log q(h|v)
        log_q = gauss_log_likelihood(h, mu, log_var)
        return log_q

    def forward(self, x, mode):
        if mode == 'inference':
            v = x
            out = self.qnet(v)
            mu = self.mu_layer(out)
            log_var = self.var_layer(out)
            return mu, log_var
        elif mode == 'sample':
            h = x
            v = self.pnet(h)
            return v
        else:
            raise NotImplementedError


class PriorVAE(nn.Module):
    def __init__(self, v_shape, h_dim, Qbasec, Pbasec):
        super(PriorVAE, self).__init__()
        assert len(v_shape) == 4, 'v_shape should be [b, c, h, w]'

        if v_shape[2] == 32:
            inc = v_shape[1]
            outc = h_dim
            self.qnet = nn.Sequential(
                nn.Conv2d(inc, Qbasec, 3, 1, 1, bias=False),
                nn.BatchNorm2d(Qbasec),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(Qbasec, Qbasec * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(Qbasec * 2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(Qbasec * 2, Qbasec * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(Qbasec * 4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(Qbasec * 4, Qbasec * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(Qbasec * 8),
                nn.LeakyReLU(0.2, inplace=True),
            )
            self.mu_layer = nn.Conv2d(Qbasec * 8, outc, 4, 1, 0)
            self.var_layer = nn.Conv2d(Qbasec * 8, outc, 4, 1, 0)

            inc = h_dim
            outc = v_shape[1]

            self.pnet = nn.Sequential(
                nn.ConvTranspose2d(inc, Pbasec * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(Pbasec * 8),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(Pbasec * 8, Pbasec * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(Pbasec * 8),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(Pbasec * 8, Pbasec * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(Pbasec * 4),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(Pbasec * 4, Pbasec * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(Pbasec * 2),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(Pbasec * 2, outc, 3, 1, 1),
                nn.Tanh()
            )
        elif v_shape[2] == 64:
            inc = v_shape[1]
            outc = h_dim
            self.qnet = nn.Sequential(
                nn.Conv2d(inc, Qbasec, 3, 1, 1, bias=False),
                nn.BatchNorm2d(Qbasec),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(Qbasec, Qbasec * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(Qbasec * 2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(Qbasec * 2, Qbasec * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(Qbasec * 4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(Qbasec * 4, Qbasec * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(Qbasec * 8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(Qbasec * 8, Qbasec * 16, 4, 2, 1, bias=False),
                nn.BatchNorm2d(Qbasec * 16),
                nn.LeakyReLU(0.2, inplace=True),
            )
            self.mu_layer = nn.Conv2d(Qbasec * 16, outc, 4, 1, 0)
            self.var_layer = nn.Conv2d(Qbasec * 16, outc, 4, 1, 0)

            inc = h_dim
            outc = v_shape[1]
            self.pnet = nn.Sequential(
                nn.ConvTranspose2d(inc, Pbasec * 16, 4, 1, 0, bias=False),
                nn.BatchNorm2d(Pbasec * 16),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(Pbasec * 16, Pbasec * 16, 4, 2, 1, bias=False),
                nn.BatchNorm2d(Pbasec * 16),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(Pbasec * 16, Pbasec * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(Pbasec * 8),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(Pbasec * 8, Pbasec * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(Pbasec * 4),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(Pbasec * 4, Pbasec * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(Pbasec * 2),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(Pbasec * 2, outc, 3, 1, 1),
                nn.Tanh()
            )
        else:
            raise NotImplementedError

    def inference(self, v):
        return self.forward(v, mode='inference')

    def sample(self, h):
        return self.forward(h, mode='sample')

    def log_q(self, h, mu, log_var):
        # compute log q(h|v)
        log_q = gauss_log_likelihood(h, mu, log_var)
        return log_q

    def forward(self, x, mode):
        if mode == 'inference':
            v = x
            out = self.qnet(v)
            mu = self.mu_layer(out)
            log_var = self.var_layer(out)
            return mu, log_var
        elif mode == 'sample':
            h = x
            v = self.pnet(h)
            return v
        else:
            raise NotImplementedError
