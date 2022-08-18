# *-* coding:utf8 *-*

import torch
import numpy as np
from datasets.data import get_cifar_dataset, get_ood_dataset
from sklearn.metrics import roc_auc_score

def get_auroc(opt, target_dataset, QP, E):
    QP.eval()
    E.eval()

    cifar_dataloader, cifar_dataset = get_cifar_dataset(opt)
    ood_dataloader, ood_dataset = get_ood_dataset(opt, target_dataset, cifar_dataset)

    # AUROC
    with torch.no_grad():
        # cifar
        cifar_labels = np.ones(len(cifar_dataset))
        cifar_scores = []
        for i, batch in enumerate(cifar_dataloader, 0):
            v = batch
            if opt.cuda:
                v = v.cuda()
            energyQ = E.energy(v).cpu().numpy()
            cifar_scores.append(- energyQ)
        cifar_scores = np.concatenate(cifar_scores)
        # ood dataset
        ood_labels = np.zeros(len(ood_dataset))
        ood_scores = []
        for i, batch in enumerate(ood_dataloader, 0):
            v = batch
            if opt.cuda:
                v = v.cuda()
            energyQ = E.energy(v).cpu().numpy()
            ood_scores.append(- energyQ)
        ood_scores = np.concatenate(ood_scores)

        ## get auroc
        y_true = np.concatenate([cifar_labels, ood_labels])
        y_scores = np.concatenate([cifar_scores, ood_scores])
        auroc = roc_auc_score(y_true, y_scores)

    QP.train()
    E.train()
    return auroc
