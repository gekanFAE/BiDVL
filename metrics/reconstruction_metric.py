# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 05:52:32 2018
Include the mse on test set for reconstruction evaluation
@author: Quan
"""

import os

import torch
from torch.autograd import Variable

import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

from torch.utils.data.sampler import SubsetRandomSampler

def giveName(iter):  # 7 digit name.
    ans = str(iter)
    return ans.zfill(7)

def prepareTestDset(dataset, imageSize, dataroot, batchSize):
    """
    return a dataloader for test dataset
    For folder type, provide a dataroot which links to the test data
    """
    workers = 2
    if dataset in ['imagenet', 'folder', 'lfw']:
        # folder dataset
        dataset = dset.ImageFolder(root=dataroot,
                                   transform=transforms.Compose([
                                       transforms.Resize(imageSize),
                                       transforms.CenterCrop(imageSize),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))
    elif dataset == 'lsun':
        dataset = dset.LSUN(db_path=dataroot, classes=['bedroom_test'],
                            transform=transforms.Compose([
                                transforms.Resize(imageSize),
                                transforms.CenterCrop(imageSize),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
    elif dataset == 'cifar10':
        dataset = dset.CIFAR10(root=dataroot, download=True, train=False,
                               transform=transforms.Compose([
                                   transforms.Resize(imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))

    elif dataset == 'SVHN':
        dataset = dset.SVHN(root=dataroot, download=True, split='test',
                               transform=transforms.Compose([
                                   transforms.Resize(imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    elif dataset == 'celeba':
        from datasets.data import SingleImagesFolderMTDataset
        import PIL
        dataset = SingleImagesFolderMTDataset(root='./data/celeba/celeba_1000_test/',
                                                  cache='./data/celeba/celeba64_1000_test.pkl',
                                                  transform=transforms.Compose([
                                                      PIL.Image.fromarray,
                                                      transforms.Resize(imageSize),
                                                      transforms.CenterCrop(imageSize),
                                                      transforms.ToTensor(),
                                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                                  ]))

    assert dataset

    dataloader = torch.utils.data.DataLoader(
        dataset, shuffle=False, batch_size=batchSize, num_workers=int(workers))

    return dataloader

def mse_score(dataloader, QP, imageSize, batchSize, saveFolder):

    input = torch.FloatTensor(batchSize, 3, imageSize, imageSize).cuda()

    total = 0
    batch_error = 0.0
    for i, batch in enumerate(dataloader, 0):
        bs = batch.shape[0]
        v = batch
        v = v.cuda()

        with torch.no_grad():
            mu, _ = QP.inference(v)
            v_recon = QP.sample(mu).cpu()

        batch_error = batch_error + torch.sum((v_recon.data - v.cpu().data)**2)
        total = total + bs

        if i % 10 ==0:
            # get the grid representation for first batch (easy to examine)
            vutils.save_image(v.data, os.path.join(saveFolder, "step_{}_input_test.png".format(i)),
                              normalize=True, nrow=10)
            vutils.save_image(v_recon.data, os.path.join(saveFolder, "step_{}_recon_test.png".format(i)),
                              normalize=True, nrow=10)

    mse = batch_error.data.item() / total

    return mse

