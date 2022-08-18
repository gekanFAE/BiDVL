# *-* coding:utf8 *-*


import torch
import numpy as np


def get_norm(net):
    param_norm = torch.sqrt(sum(torch.sum(p ** 2) for p in net.parameters()))
    grad_norm = torch.sqrt(sum(torch.sum(p.grad ** 2) for p in net.parameters()))
    return param_norm, grad_norm


def reparemetrize(mu, log_var):
    # bs * h_dim * 1 * 1
    eps = torch.randn_like(log_var)
    h = mu + torch.exp(log_var * 0.5) * eps
    return h


def gauss_log_likelihood(h, mu, log_var):
    # bs * h_dim * 1 * 1
    a = -0.5 * h.shape[1] * np.log(2 * np.pi)
    b = - (0.5 * log_var).squeeze().sum(dim=1)
    c = -0.5 * (torch.mul(h - mu, h - mu) / (1e-6 + torch.exp(log_var))).squeeze().sum(dim=1)
    ll = a + b + c
    return ll


def standard_gauss_log_likelihood(h):
    # bs * h_dim * 1 * 1
    a = -0.5 * h.shape[1] * np.log(2 * np.pi)
    # b = 0 for standard gauss
    c = -0.5 * (h ** 2).squeeze().sum(dim=1)
    ll = a + c
    return ll


def kl_regularization(mu, log_var):
    # bs * h_dim * 1 * 1
    # KL(N(mu,std)||N(0,1))
    out = -0.5 * (log_var - mu ** 2 - torch.exp(log_var) + 1.).squeeze().sum(dim=1)
    return out


def loss_analysis_kl(log_q, energy):
    # from the analytic derivation
    out = 0.5 * (log_q ** 2) + energy * log_q
    return out


def loss_kl(log_q, energy):
    out = log_q + energy
    return out


def unnormalize(img):
    return img / 2.0 + 0.5


def Xavier_init(m):
    """
    xavier initialization
    """
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        # m.weight.data.normal_()
        # nn.init.xavier_normal_(m.weight)
        m.weight.data.normal_(0.0, 0.02)
        #m.weight.data.fill_(0.001)
    #elif classname.find('Linear') != -1:
        #xavier_uniform(m.weight)
    elif classname.find('BatchNorm2d') != -1:
        # m.weight.data.normal_(1.0, 0.02)
        m.weight.data.fill_(1)
        m.bias.data.fill_(0)


def langevin_dynamic(fn, init, var, steps):
    epsilon = init
    epsilon.requires_grad_(True)

    for _ in range(steps):
        noise = torch.randn_like(epsilon)
        loss = fn(epsilon).sum()
        loss.backward()

        epsilon.grad.data.clamp_(-0.01, 0.01)
        epsilon.data.add(epsilon.grad.data, alpha=-0.5 * var)
        epsilon.data.add_(noise, alpha=np.sqrt(var))

        epsilon.grad.detach_()
        epsilon.grad.zero_()

        loss = loss.detach()
        noise = noise.detach()

    epsilon = epsilon.detach()
    return epsilon
