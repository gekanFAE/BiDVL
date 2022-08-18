# *-* coding:utf8 *-*


"""
Bi-level Doubly Variational Learning
available seeds:
"""

import os
import random
from shutil import copyfile
import datetime
import logging
import sys
import argparse

import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.utils as vutils
import higher

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from datasets.data import get_dataset
from models.inf_gen import VAE
from models.ebm import ConvE
from utils import *
import metrics.fid_v2_tf as fid_v2
from metrics.reconstruction_metric import mse_score
from metrics.auroc import get_auroc


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cifar10',  help='SVHN| cifar10 | lsun | imagenet | folder | lfw | fake')
    parser.add_argument('--dataroot', default='./data/cifar/', help='path to dataset')
    parser.add_argument('--outf', default='', help='folder to output images and model checkpoints')
    parser.add_argument('--netQP', default='', help="path to netQP (to continue training)")
    parser.add_argument('--netE', default='', help="path to netE (to continue training)")
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
    parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--gpu', type=int, default=0, metavar='S', help='gpu id (default: 0)')
    parser.add_argument('--manualSeed', default=None, type=int, help='42 is the answer to everything')

    parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
    parser.add_argument('--sampleSize', type=int, default=100, help='sample size used for generation evaluation')
    parser.add_argument('--sampleRun', type=int, default=10, help='the number of times we compute inception like score')
    parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
    parser.add_argument('--h_dim', type=int, default=100, help='size of the latent vector')
    parser.add_argument('--Qbasec', type=int, default=48)
    parser.add_argument('--Ebasec', type=int, default=64)
    parser.add_argument('--Pbasec', type=int, default=48)
    parser.add_argument('--Eactivate', default='softplus', help='tanh | sigmoid | identity | softplus')

    parser.add_argument('--epochs', type=int, default=600, help='number of epochs to train for')
    parser.add_argument('--lrE', type=float, default=0.0001, help='learning rate for E, default=0.0001')
    parser.add_argument('--lrQP', type=float, default=0.0003, help='learning rate for Q&P, default=0.0003')
    parser.add_argument('--beta1E', type=float, default=0., help='beta1 for adam E. default=0.5')
    parser.add_argument('--beta1QP', type=float, default=0., help='beta1 for adam Q&P. default=0.5')
    parser.add_argument('--lamb_vae', type=float, default=0.05, help='basic ratio between model and real density')

    parser.add_argument('--Lsteps', type=int, default=1, help='number of LL steps for each UL step')
    parser.add_argument('--Rsteps', type=int, default=0, help='number of gradient unroll steps for each UL step')
    parser.add_argument('--Edecay', type=float, default=0, help='weight decay for E')
    parser.add_argument('--QPdecay', type=float, default=1e-4, help='weight decay for Q&P')
    parser.add_argument('--Egamma', type=float, default=0.998, help='lr decay for E')
    parser.add_argument('--QPgamma', type=float, default=0.998, help='lr decay for Q&P')

    parser.add_argument('--visIter', default=1, help='number of iterations we need to visualize')
    parser.add_argument('--evalIter', default=50, help='number of iterations we need to evaluate')
    parser.add_argument('--saveIter', default=50, help='number of epochs we need to save the model')
    parser.add_argument('--diagIter', default=1, help='number of epochs we need to save the model')

    opt = parser.parse_args()
    if torch.cuda.is_available() and not opt.cuda:
        opt.cuda = True
    return opt


def set_global_gpu_env(opt):
    torch.cuda.set_device(opt.gpu)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu)


def set_seed(opt):
    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    # np.random.seed(opt.manualSeed)
    if opt.cuda:
        torch.cuda.manual_seed_all(opt.manualSeed)
        torch.backends.cudnn.deterministic = True
        cudnn.benchmark = False


def get_output_dir(exp_id, fs_prefix='../log/'):
    t = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    output_dir = os.path.join(fs_prefix + 'BiDVL/' + exp_id, t)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir


def copy_source(file, output_dir):
    copyfile(file, os.path.join(output_dir, os.path.basename(file)))


def setup_logging(output_dir):
    log_format = logging.Formatter("%(asctime)s : %(message)s")
    logger = logging.getLogger()
    logger.handlers = []
    output_file = os.path.join(output_dir, 'output.log')
    file_handler = logging.FileHandler(output_file)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)


def plot_stats(stat_1, stat_1_i, output_dir):
    p_i = 1
    p_n = len(stat_1)

    f = plt.figure(figsize=(20, p_n * 5))

    def plot(stats, stats_i):
        nonlocal p_i
        for j, (k, v) in enumerate(stats.items()):
            plt.subplot(p_n, 1, p_i)
            plt.plot(stats_i, v)
            plt.ylabel(k)
            p_i += 1

    plot(stat_1, stat_1_i)

    f.savefig(os.path.join(output_dir, 'stat.pdf'), bbox_inches='tight')
    plt.close(f)


def main():
    opt = parse_args()

    # setting
    set_global_gpu_env(opt)
    set_seed(opt)
    BS = opt.batchSize
    h_dim = opt.h_dim
    v_shape = [BS, 3, opt.imageSize, opt.imageSize]

    # log init
    exp_id = os.path.splitext(os.path.basename(__file__))[0]
    output_dir = get_output_dir(exp_id)
    copy_source(__file__, output_dir)
    setup_logging(output_dir)
    logging.info(opt)
    output_subdirs = output_dir + opt.outf
    try:
        os.makedirs(output_subdirs)
    except OSError:
        pass
    outf_recon = output_subdirs + '/recon'
    outf_syn = output_subdirs + '/syn'
    try:
        os.makedirs(outf_recon)
        os.makedirs(outf_syn)
    except OSError:
        pass

    # train dataloader
    dataloader, dataset_full, test_dataloader = get_dataset(opt)
    train_size, test_size = dataloader.__len__() * opt.batchSize, test_dataloader.__len__() * opt.batchSize

    # models
    QP = VAE(v_shape, h_dim, opt.Qbasec, opt.Pbasec)
    E = ConvE(v_shape, opt.Ebasec, opt.Eactivate)
    QP.apply(Xavier_init)
    E.apply(Xavier_init)
    criterion_MSE = nn.MSELoss(reduction='sum')
    if opt.cuda:
        QP.cuda()
        E.cuda()
        criterion_MSE.cuda()
    optimizerQP = optim.Adam(QP.parameters(), lr=opt.lrQP, weight_decay=opt.QPdecay, betas=(opt.beta1QP, 0.9))
    optimizerE = optim.Adam(E.parameters(), lr=opt.lrE, weight_decay=opt.Edecay, betas=(opt.beta1E, 0.9))
    if opt.netQP != '':
        ckpt = torch.load(opt.netQP)
        QP.load_state_dict(ckpt['model_state'])
        optimizerQP.load_state_dict(ckpt['optimizer_state'])
    print(QP)
    if opt.netE != '':
        ckpt = torch.load(opt.netE)
        E.load_state_dict(ckpt['model_state'])
        optimizerE.load_state_dict(ckpt['optimizer_state'])
    print(E)
    lr_scheduleQP = optim.lr_scheduler.ExponentialLR(optimizerQP, opt.QPgamma)
    lr_scheduleE = optim.lr_scheduler.ExponentialLR(optimizerE, opt.Egamma)

    # statics
    stats_headings = [['epoch', '{:>14}', '{:>14d}'],
                      ['lossUL', '{:>14}', '{:>14.3f}'],
                      ['lossLL_tilde', '{:>14}', '{:>14.3f}'],
                      ['norm(grad(QP))', '{:>14}', '{:>14.3f}'],
                      ['norm(grad(E))', '{:>14}', '{:>14.3f}'],
                      ['fid', '{:>14}', '{:>14.3f}'],
                      ['auroc_svhn', '{:>14}', '{:>14.3f}'],
                      ['auroc_celeba', '{:>14}', '{:>14.3f}'],
                      ['mse(val)', '{:>14}', '{:>14.3f}']]
    stat_1_i = []
    stat_1 = {
        'auroc_svhn': [],
        'auroc_celeba': [],
        'fid': [],
        'mse_score': [],
        'lossUL': [],
        'lossLL_tilde': [],
        'norm(grad(QP))': [],
        'norm(grad(E))': [],
        'norm(weight(QP))': [],
        'norm(weight(E))': [],
        'energyQ': [],
        'energyP': [],
        'loss_recon': [],
        'loss_h_recon': [],
        'norm(std)': [],
        'norm(std_hat)': [],
    }
    fid = 0.0
    mse_val = 0.0
    auroc_svhn = 0.
    auroc_celaba = 0.

    # train
    if opt.netQP != '' and opt.netE != '':
        start_epoch = torch.load(opt.netE)['epoch'] + 1
    else:
        start_epoch = 0

    h_hat = torch.FloatTensor(BS, h_dim, 1, 1)
    fixed_h = torch.FloatTensor(BS, h_dim, 1, 1).normal_()
    resample_h_hat = lambda: h_hat.resize_(BS, h_dim, 1, 1).normal_()
    if opt.cuda:
        h_hat, fixed_h = h_hat.cuda(), fixed_h.cuda()

    for epoch in range(start_epoch, opt.epochs):
        lr_scheduleQP.step(epoch=epoch)
        lr_scheduleE.step(epoch=epoch)

        stats_values = {k[0]: 0 for k in stats_headings}
        stats_values['epoch'] = epoch

        num_batch = len(dataloader.dataset) / BS
        for i, batch in enumerate(dataloader, 0):
            bs = batch.shape[0]
            v = batch
            if opt.cuda:
                v = v.cuda()
            h_hat = resample_h_hat()

            """offset"""
            # upper-level optimization
            optimizerE.zero_grad()

            v_hat = QP.sample(h_hat)
            energyQ = E.energy(v).mean()
            energyP = E.energy(v_hat.detach()).mean()

            lossUL = energyQ - energyP
            lossUL_tilde = lossUL

            lossUL_tilde.backward()
            param_normE, grad_normE = get_norm(E)
            if math.isnan(grad_normE):
                print(grad_normE.item())
                raise ValueError
            optimizerE.step()

            # lower-level optimization
            with torch.no_grad():
                energyQ_ = E.energy(v)
                ratio = energyQ_.mean() - energyQ_
                ratio = torch.clamp(torch.exp(ratio), min=0.8, max=1.2)

            for Lstep in range(opt.Lsteps):
                optimizerQP.zero_grad()

                mu, log_var = QP.inference(v)
                h = reparemetrize(mu, log_var)
                v_recon = QP.sample(h)
                v_hat = QP.sample(h_hat)
                energyP_ = E.energy(v_hat).mean()
                QP.requires_grad_(False)
                mu_hat, log_var_hat = QP.inference(v_hat)
                QP.requires_grad_(True)

                # loss_recon = torch.sum((v_recon - v)**2, dim=(1, 2, 3))
                # loss_recon = torch.mean(loss_recon * ratio)
                # loss_regular = torch.mean(kl_regularization(mu, log_var) * ratio)
                loss_recon = criterion_MSE(v_recon, v) / bs
                loss_regular = kl_regularization(mu, log_var).mean()
                loss_vae = loss_recon + loss_regular
                loss_h_recon = criterion_MSE(mu_hat, h_hat) / bs
                lossLL = energyP_ + loss_h_recon / opt.h_dim
                lossLL_tilde = lossLL + loss_vae * opt.lamb_vae

                lossLL_tilde.backward()
                param_normQP, grad_normQP = get_norm(QP)
                if math.isnan(grad_normQP):
                    print(grad_normQP.item())
                    raise ValueError
                optimizerQP.step()

            """complete"""
            # # upper-level optimization
            # optimizerE.zero_grad()
            # optimizerQP.zero_grad()
            #
            # v_hat = QP.sample(h_hat)
            # mu_hat, log_var_hat = QP.inference(v_hat.detach())
            # energyQ = E.energy(v).mean()
            # energyP = E.energy(v_hat.detach()).mean()
            # loss_regular = kl_regularization(mu_hat, log_var_hat).mean()
            #
            # lossUL = energyQ - energyP
            # lossUL_tilde = lossUL + loss_regular / opt.h_dim
            #
            # lossUL_tilde.backward()
            # param_normE, grad_normE = get_norm(E)
            # if math.isnan(grad_normE):
            #     print(grad_normE.item())
            #     raise ValueError
            # optimizerE.step()
            # optimizerQP.step()
            #
            # # lower-level optimization
            # with torch.no_grad():
            #     energyQ_ = E.energy(v)
            #     ratio = energyQ_.mean() - energyQ_
            #     ratio = torch.clamp(ratio, min=0.8, max=1.2)
            #
            # for Lstep in range(opt.Lsteps):
            #     optimizerQP.zero_grad()
            #
            #     mu, log_var = QP.inference(v)
            #     h = reparemetrize(mu, log_var)
            #     v_recon = QP.sample(h)
            #     v_hat = QP.sample(h_hat)
            #     energyP_ = E.energy(v_hat).mean()
            #     mu_hat, log_var_hat = QP.inference(v_hat)
            #     h_recon_hat = reparemetrize(mu_hat, log_var_hat)
            #
            #     # loss_recon = torch.sum((v_recon - v)**2, dim=(1, 2, 3))
            #     # loss_recon = torch.mean(loss_recon * ratio)
            #     # loss_regular = torch.mean(kl_regularization(mu, log_var) * ratio)
            #     loss_recon = criterion_MSE(v_recon, v) / bs
            #     loss_regular = kl_regularization(mu, log_var).mean()
            #     loss_vae = loss_recon + loss_regular
            #     # loss_h_recon = criterion_MSE(mu_hat, h_hat) / bs
            #     loss_h_recon = criterion_MSE(h_recon_hat, h_hat) / bs
            #     lossLL = energyP_ + loss_h_recon / opt.h_dim
            #     lossLL_tilde = lossLL + loss_vae * opt.lamb_vae
            #
            #     lossLL_tilde.backward()
            #     param_normQP, grad_normQP = get_norm(QP)
            #     if math.isnan(grad_normQP):
            #         print(grad_normQP.item())
            #         raise ValueError
            #     optimizerQP.step()

            norm_std = torch.exp(0.5 * log_var).mean()
            norm_std_hat = torch.exp(0.5 * log_var_hat).mean()

            logging.info(
                '[%3d/%3d][%3d/%3d] '
                'lossLL_tilde: %10.2f, lossUL: %10.2f, '
                'loss_vae: %10.2f, loss_recon: %10.2f, loss_regular: %10.2f, '
                'energyQ: %10.2f, energyP: %10.2f, loss_h_recon: %10.2f, '
                'std: %10.2f, std_hat: %10.2f, '
                'param_normQP: %10.2f, grad_normQP: %10.2f, '
                'param_normE: %10.2f, grad_normE: %10.2f, '
                % (epoch, opt.epochs, i, len(dataloader),
                   lossLL_tilde.data.item(), lossUL.data.item(),
                   loss_vae.data.item(), loss_recon.data.item(), loss_regular.data.item(),
                   energyQ.data.item(), energyP.data.item(), loss_h_recon.item(),
                   norm_std.data.item(), norm_std_hat.data.item(),
                   param_normQP.data.item(), grad_normQP.data.item(),
                   param_normE.data.item(), grad_normE.data.item(),
                   ))

            stats_values['lossUL'] += lossUL.data.item() / num_batch
            stats_values['lossLL_tilde'] += lossLL_tilde.data.item() / num_batch
            stats_values['norm(grad(QP))'] += grad_normQP.data.item() / num_batch
            stats_values['norm(grad(E))'] += grad_normE.data.item() / num_batch

            # diagnostics
        if (epoch + 1) % opt.diagIter == 0:
            stat_1_i.append(epoch)
            stat_1['lossUL'].append(lossUL.data.item())
            stat_1['lossLL_tilde'].append(lossLL_tilde.data.item())
            stat_1['norm(grad(QP))'].append(grad_normQP.data.item())
            stat_1['norm(grad(E))'].append(grad_normE.data.item())
            stat_1['norm(weight(QP))'].append(param_normQP.data.item())
            stat_1['norm(weight(E))'].append(param_normE.data.item())
            stat_1['energyQ'].append(energyQ.data.item())
            stat_1['energyP'].append(energyP.data.item())
            stat_1['loss_recon'].append(loss_recon.data.item())
            stat_1['loss_h_recon'].append(loss_h_recon.data.item())
            stat_1['norm(std)'].append(norm_std.data.item())
            stat_1['norm(std_hat)'].append(norm_std_hat.data.item())
            stat_1['mse_score'].append(mse_val)
            stat_1['fid'].append(fid)
            stat_1['auroc_svhn'].append(auroc_svhn)
            stat_1['auroc_celeba'].append(auroc_celaba)
            plot_stats(stat_1, stat_1_i, output_dir)

        if (epoch + 1) % opt.visIter == 0:
            with torch.no_grad():
                fixed_v = QP.sample(fixed_h)
                vutils.save_image(fixed_v.data, '%s/epoch_%03d_iter_%03d_samples_train.png' % (outf_syn, epoch, i),
                                  normalize=True, nrow=int(np.sqrt(opt.batchSize)))

                test_mu, _ = QP.inference(v)
                test_recon = QP.sample(test_mu)
                vutils.save_image(test_recon.data,
                                  '%s/epoch_%03d_iter_%03d_reconstruct_input_train.png' % (outf_recon, epoch, i),
                                  normalize=True, nrow=int(np.sqrt(opt.batchSize)))

                fixed_mu, _ = QP.inference(fixed_v)
                fixed_recon = QP.sample(fixed_mu)
                vutils.save_image(fixed_recon.data,
                                  '%s/epoch_%03d_iter_%03d_reconstruct_samples_train.png' % (outf_syn, epoch, i),
                                  normalize=True, nrow=int(np.sqrt(opt.batchSize)))

        if (epoch + 1) % opt.saveIter == 0:

            opt_dict = {'netQP': (QP, optimizerQP), 'netE': (E, optimizerE)}

            for key in opt_dict:
                save_dict = {
                    'epoch': epoch,
                    'model_state': opt_dict[key][0].state_dict(),
                    'optimizer_state': opt_dict[key][1].state_dict()
                }
                torch.save(save_dict, '%s/%s_epoch_%d.pth' % (output_dir, key, epoch))
        if (epoch + 1) % opt.evalIter == 0:
            # get the fid v2 score
            to_nhwc = lambda x: np.transpose(x, (0, 2, 3, 1))
            with torch.no_grad():
                fixed_v = torch.cat([QP.sample(resample_h_hat()).detach().cpu()
                                     for _ in range(int(train_size / opt.batchSize))])
                gen_samples_np = 255 * unnormalize(fixed_v.numpy())
                gen_samples_np = to_nhwc(gen_samples_np)
                def create_session():
                    import tensorflow as tf
                    config = tf.ConfigProto()
                    config.gpu_options.allow_growth = True
                    config.gpu_options.per_process_gpu_memory_fraction = 0.5
                    config.gpu_options.visible_device_list = str(opt.gpu)
                    return tf.Session(config=config)
                fid = fid_v2.fid_score(create_session, 255 * to_nhwc(unnormalize(dataset_full)), gen_samples_np)

            mse_val = mse_score(test_dataloader, QP, opt.imageSize, opt.batchSize, outf_recon)
            auroc_svhn = get_auroc(opt, 'svhn', QP, E)
            auroc_celaba = get_auroc(opt, 'celeba', QP, E)

            logging.info("FID:{}, MSE:{}, AUROC_SVHN:{}, AUROC_CELEBA:{}"
                         .format(fid, mse_val, auroc_svhn, auroc_celaba))

        stats_values['auroc_svhn'] = auroc_svhn
        stats_values['auroc_celeba'] = auroc_celaba
        stats_values['fid'] = fid
        stats_values['mse(val)'] = mse_val
        logging.info(''.join([h[2] for h in stats_headings]).format(*[stats_values[k[0]] for k in stats_headings]))


if __name__ == '__main__':
    main()
