from __future__ import print_function

import argparse, os, shutil, time, random, math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torch.nn.functional as F

import losses

from datasets.cifar100 import *

from train.train import *
from train.validate import *

from models.net import *
from losses.loss import *

from utils.config import *
from utils.plot import *
from utils.common import make_imb_data, save_checkpoint, hms_string

from utils.logger import logger

args = parse_args()
reproducibility(args.seed)
args = dataset_argument(args)
args.logger = logger(args)

best_acc = 0


def main():
    global best_acc

    try:
        assert args.num_max <= 50000. / args.num_class
    except AssertionError:
        args.num_max = int(50000 / args.num_class)
    
    print(f'==> Preparing imbalanced CIFAR-100')
    # N_SAMPLES_PER_CLASS = make_imb_data(args.num_max, args.num_class, args.imb_ratio)
    trainset, testset = get_cifar100(os.path.join(args.data_dir, 'cifar100/'), args)
    N_SAMPLES_PER_CLASS = trainset.img_num_list
    print("img num list : " , trainset.img_num_list)
    trainloader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last= False, pin_memory=True, sampler=None)
    testloader = data.DataLoader(testset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True) 
    
    if args.cmo:
        cls_num_list = N_SAMPLES_PER_CLASS
        cls_weight = 1.0 / (np.array(cls_num_list))
        cls_weight = cls_weight / np.sum(cls_weight) * len(cls_num_list)
        labels = trainloader.dataset.targets
        samples_weight = np.array([cls_weight[t] for t in labels])
        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.double()
        print("samples_weight", samples_weight)
        sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(labels), replacement=True)
        weighted_trainloader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, sampler=sampler)
    else:
        weighted_trainloader = None
    

    # Model
    print ("==> creating {}".format(args.network))
    model = get_model(args, N_SAMPLES_PER_CLASS)
    train_criterion = get_loss(args, N_SAMPLES_PER_CLASS)
    criterion = nn.CrossEntropyLoss() # For test, validation 
    optimizer = get_optimizer(args, model)
    scheduler = get_scheduler(args,optimizer)

    teacher = load_model(args)


    train = get_train_fn(args)
    validate = get_valid_fn(args)
    
    start_time = time.time()
    
    test_accs = []
    weight_0 = []
    weight_5 = []
    weight_10 = []
    weight_20 = []
    weight_40 = []
    weight_60 = []
    weight_80 = []
    for epoch in range(args.epochs):
        
        lr = adjust_learning_rate(optimizer, epoch, scheduler, args)
        trainloader.dataset.update_aug()
        train_loss = train(args, trainloader, model, optimizer,train_criterion, epoch, weighted_trainloader, teacher) 

        test_loss, test_acc, test_cls = validate(args, testloader, model, criterion, N_SAMPLES_PER_CLASS,  num_class=args.num_class, mode='test Valid')

        if best_acc <= test_acc:
            best_acc = test_acc
            many_best = test_cls[0]
            med_best = test_cls[1]
            few_best = test_cls[2]
            # Save models
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model['model'].state_dict() if args.loss_fn == 'ncl' else model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, epoch + 1, args.out)
        test_accs.append(test_acc)

        args.logger(f'Epoch: [{epoch+1} | {args.epochs}]', level=1)
        args.logger(f'[Train]\tLoss:\t{train_loss:.4f}', level=2)
        args.logger(f'[Test ]\tLoss:\t{test_loss:.4f}\tAcc:\t{test_acc:.4f}', level=2)
        args.logger(f'[Stats]\tMany:\t{test_cls[0]:.4f}\tMedium:\t{test_cls[1]:.4f}\tFew:\t{test_cls[2]:.4f}', level=2)
        args.logger(f'[Best ]\tAcc:\t{np.max(test_accs):.4f}\tMany:\t{100*many_best:.4f}\tMedium:\t{100*med_best:.4f}\tFew:\t{100*few_best:.4f}', level=2)
        args.logger(f'[Param]\tLR:\t{lr:.8f}', level=2)
        args.logger(f'temp array : {trainloader.dataset.temp}' , level=2)
        args.logger(f'choose aug type : {trainloader.dataset.aug_choose}' , level=2)
        args.logger(f'weight of class 0 , 5 , 10 , 20 , 40 , 60 , 80: {trainloader.dataset.aug_weight[0]}\t{trainloader.dataset.aug_weight[5]}\t{trainloader.dataset.aug_weight[10]}\t{trainloader.dataset.aug_weight[20]}\t{trainloader.dataset.aug_weight[40]}\t{trainloader.dataset.aug_weight[60]}\t{trainloader.dataset.aug_weight[80]}\t',level=2)
        # weight_0.append(trainloader.dataset.aug_weight[0].tolist())
        # weight_5.append(trainloader.dataset.aug_weight[5].tolist())
        # weight_10.append(trainloader.dataset.aug_weight[10].tolist())
        # weight_20.append(trainloader.dataset.aug_weight[20].tolist())
        # weight_40.append(trainloader.dataset.aug_weight[40].tolist())
        # weight_60.append(trainloader.dataset.aug_weight[60].tolist())
        # weight_80.append(trainloader.dataset.aug_weight[80].tolist())
    end_time = time.time()

    # Print the final results
    args.logger(f'Final performance...', level=1)
    args.logger(f'best bAcc (test):\t{np.max(test_accs)}', level=2)
    args.logger(f'best statistics:\tMany:\t{many_best}\tMed:\t{med_best}\tFew:\t{few_best}', level=2)
    args.logger(f'Training Time: {hms_string(end_time - start_time)}', level=1)
    # args.logger(f'weight matrix of class 0 : {weight_0}' , level = 1)
    # args.logger(f'weight matrix of class 5 : {weight_5}' , level = 1)
    # args.logger(f'weight matrix of class 10 : {weight_10}' , level = 1)
    # args.logger(f'weight matrix of class 20 : {weight_20}' , level = 1)
    # args.logger(f'weight matrix of class 40 : {weight_40}' , level = 1)
    # args.logger(f'weight matrix of class 60 : {weight_60}' , level = 1)
    # args.logger(f'weight matrix of class 80 : {weight_80}' , level = 1)
    if args.verbose:
        args.logger.map_save(maps)

if __name__ == '__main__':
    main()