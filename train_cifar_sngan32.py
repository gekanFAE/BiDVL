# *-* coding:utf8 *-*


"""
Train SNGAN32 on cifar-10
"""

import os
import math
import time
import random
from shutil import copyfile
import datetime
import logging
import sys
import argparse
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.utils as vutils
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datasets.data import get_dataset
from models.res_ebm import ResEBM32
from models.res_inf_gen import ResEncoder32, ResDecoder32
from utils import *
import metrics.fid_v2_tf as fid_v2
from metrics.reconstruction_metric import mse_score
from metrics.auroc import get_auroc


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cifar10',  help='SVHN| cifar10 | lsun | imagenet | folder | lfw | fake')
    parser.add_argument('--dataroot', default='./data/cifar/', help='path to dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--manualSeed', default=None, type=int, help='42 is the answer to everything')
    parser.add_argument('--gpu', type=int, default=0, metavar='S', help='gpu id (default: 0)')

    parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
    parser.add_argument('--sampleRun', type=int, default=10, help='the number of times we compute inception like score')
    parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
    parser.add_argument('--num_steps', type=int, default=3e5, help='number of epochs to train for')
    parser.add_argument('--h_dim', type=int, default=128, help='size of the latent vector')
    parser.add_argument('--n_dis', type=int, default=5, help='number of LL steps for each UL step')
    parser.add_argument('--Lsteps', type=int, default=1, help='number of LL steps for each UL step')
    parser.add_argument('--Rsteps', type=int, default=0, help='number of gradient unroll steps for each UL step')
    parser.add_argument('--lamb_vae', type=float, default=0.05, help='basic ratio between model and real density')

    parser.add_argument('--lrE', type=float, default=0.0002, help='learning rate for E, default=0.0002')
    parser.add_argument('--lrQ', type=float, default=0.0002, help='learning rate for Q, default=0.0002')
    parser.add_argument('--lrP', type=float, default=0.0002, help='learning rate for P, default=0.0002')
    parser.add_argument('--beta1E', type=float, default=0., help='beta1 for adam E. default=0.5')
    parser.add_argument('--beta1Q', type=float, default=0., help='beta1 for adam Q. default=0.5')
    parser.add_argument('--beta1P', type=float, default=0., help='beta1 for adam P. default=0.5')
    parser.add_argument('--Edecay', type=float, default=0, help='weight decay for E')
    parser.add_argument('--Qdecay', type=float, default=0, help='weight decay for Q')
    parser.add_argument('--Pdecay', type=float, default=0, help='weight decay for P')
    parser.add_argument('--Egamma', type=float, default=0.998, help='lr exponential decay for E')
    parser.add_argument('--Qgamma', type=float, default=0.998, help='lr exponential decay for Q')
    parser.add_argument('--Pgamma', type=float, default=0.998, help='lr exponential decay for P')

    parser.add_argument('--visIter', default=1, help='number of iterations we need to visualize')
    parser.add_argument('--evalIter', default=25, help='number of iterations we need to evaluate')
    parser.add_argument('--saveIter', default=25, help='number of epochs we need to save the model')
    parser.add_argument('--diagIter', default=1, help='number of epochs we need to save the model')
    parser.add_argument('--outf', default='', help='folder to output images and model checkpoints')

    opt = parser.parse_args()
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
    set_global_gpu_env(opt)
    set_seed(opt)

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

    """train"""
    num_steps = opt.num_steps
    n_dis = opt.n_dis
    BS = opt.batchSize
    h_dim = opt.h_dim

    # dataloader
    dataloader, dataset_full, test_dataloader = get_dataset(opt)
    iters_per_epoch = dataloader.__len__()
    train_size, test_size = dataloader.__len__() * BS, test_dataloader.__len__() * BS

    # models
    E = ResEBM32().cuda()
    P = ResDecoder32().cuda()
    optimizerE = optim.Adam(E.parameters(), lr=opt.lrE, weight_decay=opt.Edecay, betas=(opt.beta1E, 0.9))
    optimizerP = optim.Adam(P.parameters(), lr=opt.lrP, weight_decay=opt.Pdecay, betas=(opt.beta1P, 0.9))
    print(E)
    print(P)
    lr_scheduleE = optim.lr_scheduler.ExponentialLR(optimizerE, opt.Egamma)
    lr_scheduleP = optim.lr_scheduler.ExponentialLR(optimizerP, opt.Pgamma)

    # train
    start_time = time.time()
    iter_dataloader = iter(dataloader)
    global_step = 0
    fixed_h = torch.randn((BS, h_dim)).cuda()
    while global_step < num_steps:
        for i in range(n_dis):
            try:
                v = next(iter_dataloader)
            except StopIteration:
                iter_dataloader = iter(dataloader)
                v = next(iter_dataloader)
            v = v.cuda()
            bs = v.shape[0]

            optimizerE.zero_grad()

            h_hat = torch.randn((bs, h_dim)).cuda()
            v_hat = P.sample(h_hat)
            energyQ = E.energy(v)
            energyP = E.energy(v_hat.detach())

            lossUL = F.relu(1.0 + energyQ).mean() + F.relu(1.0 - energyP).mean()  # hinge
            # lossUL = energyQ.mean() - energyP.mean()  # canonical NLL for EBM
            lossUL_tilde = lossUL

            lossUL_tilde.backward()
            param_normE, grad_normE = get_norm(E)
            if math.isnan(grad_normE):
                print(grad_normE.item())
                raise ValueError
            optimizerE.step()

            logging.info(
                '[%3d/%3d][%3d/%3d] '
                'lossUL: %10.2f, energyQ: %10.2f, energyP: %10.2f, '
                'param_normE: %10.2f, grad_normE: %10.2f, '
                % (global_step, num_steps, i, n_dis,
                   lossUL.data.item(), energyQ.mean().data.item(), energyP.mean().data.item(),
                   param_normE.data.item(), grad_normE.data.item(),
                   ))

            if i == (n_dis - 1):
                optimizerP.zero_grad()

                h_hat = torch.randn((bs, h_dim)).cuda()
                v_hat = P.sample(h_hat)
                energyP_ = E.energy(v_hat)

                lossLL = energyP_.mean()  # hinge
                lossLL_tilde = lossLL

                lossLL_tilde.backward()
                param_normP, grad_normP = get_norm(P)
                if math.isnan(grad_normP):
                    print(grad_normP.item())
                    raise ValueError
                optimizerP.step()

                logging.info(
                    'energyP_: %10.2f, '
                    'param_normP: %10.2f, grad_normP: %10.2f, '
                    % (energyP_.mean().data.item(),
                       param_normP.data.item(), grad_normP.data.item(),
                       ))

        global_step += 1
        if (global_step * n_dis) % iters_per_epoch == 0:
            lr_scheduleE.step(epoch=int(global_step * n_dis / iters_per_epoch))
            lr_scheduleP.step(epoch=int(global_step * n_dis / iters_per_epoch))

        # diagnostics
        if (global_step * n_dis) % (iters_per_epoch * opt.visIter) == 0:
            with torch.no_grad():
                fixed_v = P.sample(fixed_h)
                vutils.save_image(fixed_v.data, '%s/global_step_%03d_samples_train.png' % (outf_syn, global_step),
                                  normalize=True, nrow=int(np.sqrt(opt.batchSize)))

        # if (global_step * n_dis) % (iters_per_epoch * opt.saveIter) == 0:
        #     opt_dict = {'netQP': (P, optimizerP), 'netE': (E, optimizerE)}
        #     for key in opt_dict:
        #         save_dict = {'epoch': global_step,
        #                      'model_state': opt_dict[key][0].state_dict(),
        #                      'optimizer_state': opt_dict[key][1].state_dict()}
        #         torch.save(save_dict, '%s/%s_epoch_%d.pth' % (output_dir, key, global_step))

        if (global_step * n_dis) % (iters_per_epoch * opt.evalIter) == 0:
            # get the fid v2 score
            to_nhwc = lambda x: np.transpose(x, (0, 2, 3, 1))
            with torch.no_grad():
                h_hat = torch.randn((100, h_dim)).cuda()
                resample_h_hat = lambda: h_hat.resize_(100, h_dim).normal_()
                fixed_v = torch.cat([P.sample(resample_h_hat()).detach().cpu()
                                     for _ in range(500)])
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
            logging.info("FID:{}"
                         .format(fid))

    end_time = time.time()
    logging.info("train_time:{}".format(end_time-start_time))


if __name__ == '__main__':
    main()
