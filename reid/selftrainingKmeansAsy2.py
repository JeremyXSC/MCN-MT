#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function, absolute_import
import argparse
import time
import os.path as osp
import os 
import numpy as np
import torch
import csv
import codecs
from torch import nn
from torch.nn import init
from torch.backends import cudnn
from torch.utils.data import DataLoader
from reid import datasets
from reid import models
from reid.dist_metric import DistanceMetric
from reid.loss import TripletLoss
from reid.trainers import Trainer, CoTrainerAsy, CoTeaching, CoTrainerAsySep
from reid.evaluators import Evaluator, extract_features
from reid.utils.data import transforms as T
import torch.nn.functional as F
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.data.sampler import RandomIdentitySampler
from reid.utils.serialization import load_checkpoint, save_checkpoint

from sklearn.cluster import KMeans
from reid.rerank import re_ranking



def data_write_csv(file_name, datas):#file_name为写入CSV文件的路径，datas为要写入数据列表
  file_csv = codecs.open(file_name,'w+','utf-8')#追加
  writer = csv.writer(file_csv, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
  for data in datas:
    writer.writerow(data)
  print("保存文件成功，处理结束")


def get_data(name, data_dir, height, width, batch_size,
             workers):
    root = osp.join(data_dir, name)

    dataset = datasets.create(name, root, num_val=0.1)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    # use all training and validation images in target dataset
    train_set = dataset.trainval 
    num_classes = dataset.num_trainval_ids

    transformer = T.Compose([
        T.Resize((height,width)),
        T.ToTensor(),
        normalizer,
    ])

    extfeat_loader = DataLoader(
        Preprocessor(train_set, root=dataset.images_dir,
                     transform=transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    test_loader = DataLoader(
        Preprocessor(list(set(dataset.query) | set(dataset.gallery)),
                     root=dataset.images_dir, transform=transformer),
        batch_size=batch_size//2, num_workers=workers,
        shuffle=False, pin_memory=True)

    return dataset, num_classes, extfeat_loader, test_loader


def get_source_data(name, data_dir, height, width, batch_size, workers):
    root = osp.join(data_dir, name)

    dataset = datasets.create(name, root, num_val=0.1)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    # use all training images on source dataset
    train_set = dataset.train
    num_classes = dataset.num_train_ids

    transformer = T.Compose([
        T.Resize((height,width)),
        T.ToTensor(),
        normalizer,
    ])

    extfeat_loader = DataLoader(
        Preprocessor(train_set, root=dataset.images_dir,
                     transform=transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return dataset, extfeat_loader


def splitLowconfi(feature, labels, centers, ratio=0.2):
    # set bot 20% imsimilar samples to -1 
    # center VS feature
    centerDis = calDis(torch.from_numpy(feature), torch.from_numpy(centers)).numpy() # center VS samples
    noiseLoc = []
    for ii, pid in enumerate(set(labels)):
        curDis = centerDis[:,ii]
        curDis[labels!=pid] = 100
        smallLossIdx = curDis.argsort()
        smallLossIdx = smallLossIdx[curDis[smallLossIdx]!=100]
        # bot 20% removed
        partSize = int(ratio*smallLossIdx.shape[0])
        if partSize!=0:
            noiseLoc.extend(smallLossIdx[-partSize:])
    labels[noiseLoc] = -1
    return labels


def calDis(qFeature, gFeature):#246s
    x, y = F.normalize(qFeature), F.normalize(gFeature)
    # x, y = qFeature, gFeature
    m, n = x.shape[0], y.shape[0]
    disMat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
             torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    disMat.addmm_(1, -2, x, y.t())
    return disMat.clamp_(min=1e-5)


def labelUnknown(knownFeat, allLab, unknownFeat):
    disMat = calDis(knownFeat, unknownFeat)
    labLoc = disMat.argmin(dim=0)
    return allLab[labLoc]


def labelNoise(feature, labels):
    # features and labels with -1
    noiseFeat, pureFeat = feature[labels==-1,:], feature[labels!=-1,:]
    labels = labels[labels!=-1]
    unLab = labelUnknown(pureFeat, labels, noiseFeat)
    return unLab.numpy()


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.benchmark = True

    # Create data loaders
    assert args.num_instances > 1, "num_instances should be greater than 1"
    assert args.batch_size % args.num_instances == 0, \
        'num_instances should divide batch_size'
    if args.height is None or args.width is None:
        args.height, args.width = (144, 56) if args.arch == 'inception' else \
                                  (256, 128)

    # get source data
    src_dataset, src_extfeat_loader = \
        get_source_data(args.src_dataset, args.data_dir, args.height,
                        args.width, args.batch_size, args.workers)
    # get target data
    tgt_dataset, num_classes, tgt_extfeat_loader, test_loader = \
        get_data(args.tgt_dataset, args.data_dir, args.height,
                 args.width, args.batch_size, args.workers)

    # Create model
    # Hacking here to let the classifier be the number of source ids
    if args.src_dataset == 'dukemtmc':
        model = models.create(args.arch, num_classes=632, pretrained=False)
        coModel = models.create(args.arch, num_classes=632, pretrained=False)
    elif args.src_dataset == 'market1501':
        model = models.create(args.arch, num_classes=676, pretrained=False)
        coModel = models.create(args.arch, num_classes=676, pretrained=False)
    else:
        raise RuntimeError('Please specify the number of classes (ids) of the network.')

    # Load from checkpoint
    if args.resume:
        print('Resuming checkpoints from finetuned model on another dataset...\n')
        checkpoint = load_checkpoint(args.resume)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        coModel.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        raise RuntimeWarning('Not using a pre-trained model.')
    model = nn.DataParallel(model).cuda()
    coModel = nn.DataParallel(coModel).cuda()

    # evaluator.evaluate(test_loader, tgt_dataset.query, tgt_dataset.gallery)
    # if args.evaluate: return

    # Criterion
    criterion = [
        TripletLoss(args.margin, args.num_instances, isAvg=False, use_semi=False).cuda(),
        TripletLoss(args.margin, args.num_instances, isAvg=False, use_semi=False).cuda(),
    ]

    
    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr = args.lr
    )
    coOptimizer = torch.optim.Adam(
        coModel.parameters(), lr = args.lr
    )
    
    optims = [optimizer, coOptimizer]


    # training stage transformer on input images
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transformer = T.Compose([
        T.Resize((args.height,args.width)),
        T.RandomHorizontalFlip(),
        T.ToTensor(), normalizer,
        T.RandomErasing(probability=0.5, sh=0.2, r1=0.3)
    ])


    # # Start training
    for iter_n in range(args.iteration):
        if args.lambda_value == 0:
            source_features = 0
        else:
            # get source datas' feature
            source_features, _ = extract_features(model, src_extfeat_loader, print_freq=args.print_freq)
            # synchronization feature order with src_dataset.train
            source_features = torch.cat([source_features[f].unsqueeze(0) for f, _, _ in src_dataset.train], 0) 

        # extract training images' features
        print('Iteration {}: Extracting Target Dataset Features...'.format(iter_n+1))
        target_features, tarNames = extract_features(model, tgt_extfeat_loader, print_freq=args.print_freq)
        print('target_features.type is {}'.format(type(target_features)))
        #target_features.type is <class 'collections.OrderedDict'>
        # 先创建并打开一个文本文件
        file = open('target_features.txt', 'w') 

        # 遍历字典的元素，将每项元素的key和value分拆组成字符串，注意添加分隔符和换行符
        for k,v in target_features.items():
            file.write(str(k)+' '+str(v)+'\n')
	
        # 注意关闭文件
        file.close()


        # synchronization feature order with dataset.train
        target_features = torch.cat([target_features[f].unsqueeze(0) for f, _, _ in tgt_dataset.trainval], 0) 
        target_real_label = np.asarray([tarNames[f].unsqueeze(0) for f, _, _ in tgt_dataset.trainval]) 
        numTarID = len(set(target_real_label))
        # calculate distance and rerank result
        print('Calculating feature distances...') 
        target_features = target_features.numpy()
        np.savetxt('target_features.numpy.csv', np.array(target_features), delimiter=',', fmt='%f')
        cluster = KMeans(n_clusters=numTarID, n_jobs=8, n_init=1)

        # select & cluster images as training set of this epochs
        print('Clustering and labeling...')
        clusterRes = cluster.fit(target_features)
        labels, centers = clusterRes.labels_, clusterRes.cluster_centers_
        labels = splitLowconfi(target_features,labels,centers)
        # num_ids = len(set(labels))
        # print('Iteration {} have {} training ids'.format(iter_n+1, num_ids))
        # generate new dataset
        new_dataset, unknown_dataset = [], []
        # assign label for target ones
        unknownLab = labelNoise(torch.from_numpy(target_features), torch.from_numpy(labels))
        # unknownFeats = target_features[labels==-1,:]
        unCounter = 0
        for (fname, _, cam), label in zip(tgt_dataset.trainval, labels):
            if label==-1: 
                unknown_dataset.append((fname,int(unknownLab[unCounter]),cam)) # unknown data
                unCounter += 1
                continue
            # dont need to change codes in trainer.py _parsing_input function and sampler function after add 0
            new_dataset.append((fname,label,cam)) 
        print('Iteration {} have {} training images'.format(iter_n+1, len(new_dataset)))
        print('Iteration {} have {} outliers training images'.format(iter_n+1, len(unknown_dataset)))
        #Iteration 1 have 10643 training images
        #Iteration 1 have 2293 outliers training images

        train_loader = DataLoader(
            Preprocessor(new_dataset, root=tgt_dataset.images_dir, transform=train_transformer),
            batch_size=args.batch_size, num_workers=4,
            sampler=RandomIdentitySampler(new_dataset, args.num_instances),
            pin_memory=True, drop_last=True
        )
        # hard samples
        unLoader = DataLoader(
            Preprocessor(unknown_dataset, root=tgt_dataset.images_dir, transform=train_transformer),
            batch_size=args.batch_size, num_workers=4,
            sampler=RandomIdentitySampler(unknown_dataset, args.num_instances),
            pin_memory=True, drop_last=True
        )
        
        #*********************************二次聚类**************************************
        print('***************second cluster*************')
        print('tgt_dataset.trainval type:{}'.format(type(tgt_dataset.trainval)))
        print('new_dataset type:{}'.format(type(new_dataset)))
        #tgt_dataset.trainval type:<class 'list'>
        #new_dataset type:<class 'list'>
        
        data_write_csv('tgt_dataset.trainval.csv',tgt_dataset.trainval)
        data_write_csv('new_dataset.csv',new_dataset)


        target_features2, tarNames2 = extract_features(model, train_loader, print_freq=args.print_freq)
        # 先创建并打开一个文本文件
        file = open('target_features2.txt', 'w') 

        # 遍历字典的元素，将每项元素的key和value分拆组成字符串，注意添加分隔符和换行符
        for k,v in target_features2.items():
            file.write(str(k)+' '+str(v)+'\n')
	
        # 注意关闭文件
        file.close()
        # synchronization feature order with dataset.train
        target_features2 = torch.cat([target_features2[f].unsqueeze(0) for f, _, _ in new_dataset], 0) #需要比较tgt_dataset.trainval和new_dataset两个数据集的格式问题，格式问题导致了dict的keyerror
        target_real_label2 = np.asarray([tarNames2[f].unsqueeze(0) for f, _, _ in new_dataset]) 
        numTarID2 = len(set(target_real_label2))
        # calculate distance and rerank result
        print('Calculating feature distances...') 
        target_features2 = target_features2.numpy()
        cluster2 = KMeans(n_clusters=numTarID2, n_jobs=8, n_init=1)

        # select & cluster images as training set of this epochs
        print('Clustering and labeling...')
        clusterRes2 = cluster2.fit(target_features2)
        labels2, centers2 = clusterRes2.labels_, clusterRes2.cluster_centers_
        labels2 = splitLowconfi(target_features2,labels2,centers2,ratio=0.3)
        # num_ids = len(set(labels))
        # print('Iteration {} have {} training ids'.format(iter_n+1, num_ids))
        # generate new dataset
        new_dataset2, unknown_dataset2 = [], []
        # assign label for target ones
        unknownLab2 = labelNoise(torch.from_numpy(target_features2), torch.from_numpy(labels2))
        # unknownFeats = target_features[labels==-1,:]
        unCounter = 0
        for (fname, _, cam), label in zip(new_dataset, labels2):
            if label==-1: 
                unknown_dataset2.append((fname,int(unknownLab2[unCounter]),cam)) # unknown data
                unCounter += 1
                continue
            # dont need to change codes in trainer.py _parsing_input function and sampler function after add 0
            new_dataset2.append((fname,label,cam)) 
        print('Iteration {} have {} inliers2 training images'.format(iter_n+1, len(new_dataset2)))
        print('Iteration {} have {} outliers2 training images'.format(iter_n+1, len(unknown_dataset2)))

        train_inliers_loader = DataLoader(
            Preprocessor(new_dataset2, root=tgt_dataset.images_dir, transform=train_transformer),
            batch_size=args.batch_size, num_workers=4,
            sampler=RandomIdentitySampler(new_dataset2, args.num_instances),
            pin_memory=True, drop_last=True
        )
        # hard samples
        train_outliers_loader = DataLoader(
            Preprocessor(unknown_dataset2, root=tgt_dataset.images_dir, transform=train_transformer),
            batch_size=args.batch_size, num_workers=4,
            sampler=RandomIdentitySampler(unknown_dataset2, args.num_instances),
            pin_memory=True, drop_last=True
        )
        
        
        
        
        
        
        
        
        #**************************************************************************

        # train model with new generated dataset
        trainer = CoTrainerAsy(
            model, coModel, train_loader, unLoader, criterion, optims
        )
        # trainer = CoTeaching(
        #    model, coModel, train_loader, unLoader, criterion, optims
        # )
        # trainer = CoTrainerAsySep(
        #     model, coModel, train_loader, unLoader, criterion, optims
        # )
        
        evaluator = Evaluator(model, print_freq=args.print_freq)
        #evaluatorB = Evaluator(coModel, print_freq=args.print_freq)
        # Start training
        for epoch in range(args.epochs):
            trainer.train(epoch, remRate=0.2+(0.6/args.iteration)*(1+iter_n)) # to at most 80%
            # trainer.train(epoch, remRate=0.7+(0.3/args.iteration)*(1+iter_n))
        # test only
        rank_score = evaluator.evaluate(test_loader, tgt_dataset.query, tgt_dataset.gallery)
        #print('co-model:\n')
        #rank_score = evaluatorB.evaluate(test_loader, tgt_dataset.query, tgt_dataset.gallery)

    # Evaluate
    rank_score = evaluator.evaluate(test_loader, tgt_dataset.query, tgt_dataset.gallery)
    save_checkpoint({
                'state_dict': model.module.state_dict(),
                'epoch': epoch + 1, 'best_top1': rank_score.market1501[0],
        }, True, fpath=osp.join(args.logs_dir, 'adapted.pth.tar'))
    return (rank_score.map, rank_score.market1501[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Triplet loss classification")
    # data
    parser.add_argument('--src_dataset', type=str, default='dukemtmc',
                        choices=datasets.names())
    parser.add_argument('--tgt_dataset', type=str, default='market1501',
                        choices=datasets.names())
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--noiseLam', type=float, default=0.5)
    parser.add_argument('--height', type=int,
                        help="input height, default: 256 for resnet*, "
                             "144 for inception")
    parser.add_argument('--width', type=int,
                        help="input width, default: 128 for resnet*, "
                             "56 for inception")
    parser.add_argument('--combine-trainval', action='store_true',
                        help="train and val sets together for training, "
                             "val set alone for validation")
    parser.add_argument('--num_instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 4")
    # model
    parser.add_argument('--arch', type=str, default='resnet50',
                        choices=models.names())
    # loss
    parser.add_argument('--margin', type=float, default=0.5,
                        help="margin of the triplet loss, default: 0.5")
    parser.add_argument('--lambda_value', type=float, default=0.1,
                        help="balancing parameter, default: 0.1")
    parser.add_argument('--rho', type=float, default=1.6e-3,
                        help="rho percentage, default: 1.6e-3")
    # optimizer
    parser.add_argument('--lr', type=float, default=6e-5,
                        help="learning rate of all parameters")
    # training configs
    parser.add_argument('--resume', type=str, metavar='PATH',
                        default = '')
    parser.add_argument('--evaluate', type=int, default=0,
                        help="evaluation only")
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print_freq', type=int, default=1)
    parser.add_argument('--iteration', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=30)
    # metric learning
    parser.add_argument('--dist_metric', type=str, default='euclidean',
                        choices=['euclidean', 'kissme'])
    # misc
    parser.add_argument('--data_dir', type=str, metavar='PATH',
                        default='')
    parser.add_argument('--logs_dir', type=str, metavar='PATH',
                        default='')

    args = parser.parse_args()
    mean_ap, rank1 = main(args)
    results_file = np.asarray([mean_ap, rank1])
    file_name = time.strftime("%H%M%S", time.localtime())
    file_name = osp.join(args.logs_dir, file_name)
    np.save(file_name, results_file)
