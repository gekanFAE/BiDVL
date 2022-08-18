# author: enijkamp@ucla.edu

import os
import pickle
import numpy as np
import torch.utils.data
from torchvision import datasets, transforms
from torch.utils import data
import PIL


class UniformDataset(data.Dataset):
    def __init__(self, imageSize, nc, len):
        self.imageSize = imageSize
        self.nc = nc
        self.len = len

    def __len__(self):
        return self.len

    def __getitem__(self, _):
        X = torch.zeros(self.nc, self.imageSize, self.imageSize).uniform_(-1, 1)
        return X


class ConstantDataset(data.Dataset):
    def __init__(self, imageSize, nc, len):
        self.imageSize = imageSize
        self.nc = nc
        self.len = len

    def __len__(self):
        return self.len

    def __getitem__(self, i):
        n = torch.FloatTensor(1).uniform_(-1,1)
        X = n * torch.ones(self.nc, self.imageSize, self.imageSize)
        return X


class DTDDataset(datasets.ImageFolder):
    def __init__(self, imageSize, *args, **kwargs):
        super(DTDDataset, self).__init__(*args, **kwargs)
        self.imageSize = imageSize

    def __getitem__(self, index):
        data = PIL.Image.open(self.imgs[index][0])
        transoform_data = self.transform(data)[:,:self.imageSize,:self.imageSize]
        return transoform_data + torch.FloatTensor(transoform_data.shape).uniform_(-1/512, 1/512)


class SingleImagesFolderMTDataset(torch.utils.data.Dataset):
    def __init__(self, root, cache, transform=None, workers=32, protocol=None):
        if cache is not None and os.path.exists(cache):
            with open(cache, 'rb') as f:
                self.images = pickle.load(f)
        else:
            self.transform = transform if not transform is None else lambda x: x
            self.images = []

            def split_seq(seq, size):
                newseq = []
                splitsize = 1.0 / size * len(seq)
                for i in range(size):
                    newseq.append(seq[int(round(i * splitsize)):int(round((i + 1) * splitsize))])
                return newseq

            def map(path_imgs):
                imgs_0 = [self.transform(np.array(PIL.Image.open(os.path.join(root, p_i)))) for p_i in path_imgs]
                imgs_1 = [self.compress(img) for img in imgs_0]

                print('.')
                return imgs_1

            path_imgs = os.listdir(root)
            n_splits = len(path_imgs) // 1000
            path_imgs_splits = split_seq(path_imgs, n_splits)

            from multiprocessing.dummy import Pool as ThreadPool
            pool = ThreadPool(workers)
            results = pool.map(map, path_imgs_splits)
            pool.close()
            pool.join()

            for r in results:
                self.images.extend(r)

            if cache is not None:
                with open(cache, 'wb') as f:
                    pickle.dump(self.images, f, protocol=protocol)

        print('Total number of images {}'.format(len(self.images)))

    def __getitem__(self, item):
        return self.decompress(self.images[item])

    def __len__(self):
        return len(self.images)

    @staticmethod
    def compress(img):
        return img

    @staticmethod
    def decompress(output):
        return output


class IgnoreLabelDataset(torch.utils.data.Dataset):
    def __init__(self, orig):
        self.orig = orig

    def __getitem__(self, index):
        return self.orig[index][0]

    def __len__(self):
        return len(self.orig)


def get_dataset(opt):
    if opt.dataset == 'cifar10':
        dataset = IgnoreLabelDataset(datasets.CIFAR10(root=opt.dataroot, download=True,
                                                      transform=transforms.Compose([
                                                          transforms.Resize(opt.imageSize),
                                                          transforms.ToTensor(),
                                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                                      ])))
        test_dataset = IgnoreLabelDataset(datasets.CIFAR10(root=opt.dataroot, download=True, train=False,
                                                           transform=transforms.Compose([
                                                               transforms.Resize(opt.imageSize),
                                                               transforms.ToTensor(),
                                                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                                           ])))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                                 shuffle=True, num_workers=int(opt.workers))
        dataset_full = np.array([x.cpu().numpy() for x in iter(dataset)])
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, shuffle=False, batch_size=opt.batchSize, num_workers=int(opt.workers))

    elif opt.dataset == 'lsun':
        dataset = IgnoreLabelDataset(datasets.LSUN(root=opt.dataroot, classes='bedroom_train',
                                                   transform=transforms.Compose([
                                                       transforms.Resize(opt.imageSize),
                                                       transforms.ToTensor(),
                                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                                   ])))
        test_dataset = IgnoreLabelDataset(datasets.CIFAR10(root=opt.dataroot, download=True, train=False,
                                                           transform=transforms.Compose([
                                                               transforms.Resize(opt.imageSize),
                                                               transforms.ToTensor(),
                                                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                                           ])))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                                 shuffle=True, num_workers=int(opt.workers))
        dataset_full = np.array([x.cpu().numpy() for x in iter(dataset)])
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, shuffle=False, batch_size=opt.batchSize, num_workers=int(opt.workers))

    elif opt.dataset == 'celeba':
        dataset = IgnoreLabelDataset(datasets.ImageFolder(
            root=opt.dataroot,
            transform=transforms.Compose([
                transforms.CenterCrop(160),
                transforms.Resize(opt.imageSize),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])))
        train_size = 50000
        test_size = 1000
        dataset, test_dataset, _ = torch.utils.data.random_split(
            dataset, [train_size, test_size, len(dataset)-train_size-test_size])
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                                 shuffle=True, num_workers=int(opt.workers))
        dataset_full = np.array([x.cpu().numpy() for x in iter(dataset)])
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, shuffle=False, batch_size=opt.batchSize, num_workers=int(opt.workers))

    elif opt.dataset == 'svhn':
        dataset = IgnoreLabelDataset(datasets.SVHN(root='data/svhn/', download=True, split='train',
                                                   transform=transforms.Compose([
                                                       transforms.Resize(opt.imageSize),
                                                       transforms.ToTensor(),
                                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                                   ])))
        test_dataset = IgnoreLabelDataset(datasets.SVHN(root='data/svhn/', download=True, split='test',
                                                   transform=transforms.Compose([
                                                       transforms.Resize(opt.imageSize),
                                                       transforms.ToTensor(),
                                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                                   ])))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                                 shuffle=True, num_workers=int(opt.workers))
        dataset_full = np.array([x.cpu().numpy() for x in iter(dataset)])
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, shuffle=False, batch_size=opt.batchSize, num_workers=int(opt.workers))

    else:
        raise NotImplementedError

    return dataloader, dataset_full, test_dataloader


def get_cifar_dataset(opt):
    dataset = IgnoreLabelDataset(datasets.CIFAR10(root='data/cifar/', train=False,
                                                  transform=transforms.Compose([
                                                      transforms.Resize(opt.imageSize),
                                                      transforms.ToTensor(),
                                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                                  ])))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                             shuffle=True, num_workers=int(opt.workers))
    return dataloader, dataset


def get_ood_dataset(opt, target_dataset, cifar_dataset):
    length = len(cifar_dataset)

    if target_dataset == 'svhn':
        dataset = IgnoreLabelDataset(datasets.SVHN(root='data/svhn/', download=True, split='test',
                                     transform=transforms.Compose([
                                         transforms.Resize(opt.imageSize),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                     ])))
    elif target_dataset == 'cifar10_train':
        dataset = IgnoreLabelDataset(datasets.CIFAR10(root='data/cifar/', train=True,
                                                      transform=transforms.Compose([
                                                          transforms.Resize(opt.imageSize),
                                                          transforms.ToTensor(),
                                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                                      ])))
    elif target_dataset == 'random':
        dataset = UniformDataset(opt.imageSize, 3, length)
    elif target_dataset == 'constant':
        dataset = ConstantDataset(opt.imageSize, 3, length)
    elif target_dataset == 'texture':
        dataset = DTDDataset('data/dtd/images/',
                             transform=transforms.Compose([
                                 transforms.Resize(opt.imageSize),
                                 transforms.CenterCrop(opt.imageSize),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                             ]))
    elif target_dataset == 'celeba':
        dataset = IgnoreLabelDataset(datasets.ImageFolder(
            root='data/CelebA/Img',
            transform=transforms.Compose([
                transforms.CenterCrop(160),
                transforms.Resize(opt.imageSize),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])))
        test_size = 10000
        dataset, _ = torch.utils.data.random_split(
            dataset, [test_size, len(dataset) - test_size])
    else:
        raise ValueError('no dataset')

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                             shuffle=True, num_workers=int(opt.workers))
    return dataloader, dataset
