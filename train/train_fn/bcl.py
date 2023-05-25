from __future__ import print_function

import time
import numpy as np
import torch
import torch.nn as nn

from aug.cutmix import *

from utils.accuracy import AverageMeter
from utils.common import Bar

import copy, time
import random

from datasets.cifar100 import test_CIFAR100

def train_bcl(args, trainloader, model, optimizer, criterion, epoch, weighted_trainloader, teacher = None):
    model.train()
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()
    
    # cls_num need to get
    cls_num = trainloader.dataset.cls_num
    # correct_all = torch.zeros(cls_num).int()
    output = []
    targets = []

    bar = Bar('Training', max=len(trainloader))
        
    for batch_idx, data_tuple in enumerate(trainloader):
        inputs_b = data_tuple[0]
        targets_b = data_tuple[1]
        indexs = data_tuple[2]
        targets += targets_b.tolist()

        # Measure data loading
        data_time.update(time.time() - end)
        batch_size = targets_b.size(0)
        
        if args.cmo:
            raise "BCL not implemented for CMO..."
        else:
            inputs_b = torch.cat([inputs_b[0], inputs_b[1], inputs_b[2]], dim=0).cuda()
            batch_size = targets_b.shape[0]
            targets_b = targets_b.cuda()
            feat_mlp, logits, centers = model(inputs_b)
            centers = centers[:args.num_class]
            _, f2, f3 = torch.split(feat_mlp, [batch_size, batch_size, batch_size], dim=0)
            features = torch.cat([f2.unsqueeze(1), f3.unsqueeze(1)], dim=1)
            logits, _, __ = torch.split(logits, [batch_size, batch_size, batch_size], dim=0)
            loss = criterion(centers, logits, features, targets_b)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output += logits.max(dim=1)[1].tolist()

        # record
        losses.update(loss.item(), targets_b.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        
        # plot
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | ' \
                      'Loss: {loss:.4f}'.format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    )
        bar.next()
    bar.finish()
    if args.doda:
        for cidx in range(cls_num):
            class_pos = torch.where(torch.tensor(targets) == cidx)[0]
            correct_all = (torch.tensor(output) == torch.tensor(targets))[class_pos].int().sum()
            if epoch > 50 and correct_all > trainloader.dataset.temp[cidx]:
                trainloader.dataset.aug_weight[cidx][trainloader.dataset.aug_choose[cidx]] += 1
                # if trainloader.dataset.aug_weight[cidx][trainloader.dataset.aug_choose[cidx]] > 10:
                #     trainloader.dataset.aug_weight[cidx][trainloader.dataset.aug_choose[cidx]] -= 1
            elif epoch > 50 and correct_all < trainloader.dataset.temp[cidx]:
                trainloader.dataset.aug_weight[cidx][trainloader.dataset.aug_choose[cidx]] -= 1
                if trainloader.dataset.aug_weight[cidx][trainloader.dataset.aug_choose[cidx]] < 1:
                    trainloader.dataset.aug_weight[cidx][trainloader.dataset.aug_choose[cidx]] = 1
            trainloader.dataset.temp[cidx] = correct_all

    return losses.avg
