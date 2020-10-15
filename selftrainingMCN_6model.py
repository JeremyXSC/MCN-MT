#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function, absolute_import
import argparse
import time
import os.path as osp
import os
import sys
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
from reid.loss import TripletLoss
from reid.trainers import CoTrainerAsy, CoTrainerAsy4, CoTrainerAsy5, CoTrainerAsy6
from reid.evaluators import Evaluator, extract_features
from reid.utils.data import transforms as T
import torch.nn.functional as F
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.data.sampler import RandomIdentitySampler
from reid.utils.serialization import load_checkpoint, save_checkpoint

from sklearn.cluster import DBSCAN
from reid.rerank import re_ranking


class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


sys.stdout = Logger("./LOG/MCN_6/D2M/print.txt")

def data_write_csv(file_name, datas):#file_name为写入CSV文件的路径，datas为要写入数据列表
  file_csv = codecs.open(file_name,'w+','utf-8')#追加
  writer = csv.writer(file_csv, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
  for data in datas:
    writer.writerow(data)
  print("保存文件成功，处理结束")

def calScores(clusters, labels):
    """
    compute pair-wise precision pair-wise recall
    """
    from scipy.special import comb
    if len(clusters) == 0:
        return 0, 0
    else:
        curCluster = []
        for curClus in clusters.values():
            curCluster.append(labels[curClus])
        TPandFP = sum([comb(len(val), 2) for val in curCluster])
        TP = 0
        for clusterVal in curCluster:
            for setMember in set(clusterVal):
                if sum(clusterVal == setMember) < 2: continue
                TP += comb(sum(clusterVal == setMember), 2)
        FP = TPandFP - TP
        # FN and TN
        TPandFN = sum([comb(labels.tolist().count(val), 2) for val in set(labels)])
        FN = TPandFN - TP
        # cal precision and recall
        precision, recall = TP / (TP + FP), TP / (TP + FN)
        fScore = 2 * precision * recall / (precision + recall)
        return precision, recall, fScore


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
        T.Resize((height, width)),
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
        batch_size=batch_size // 2, num_workers=workers,
        shuffle=False, pin_memory=True)

    return dataset, num_classes, extfeat_loader, test_loader


def saveAll(nameList, rootDir, tarDir):
    import os
    import shutil
    if os.path.exists(tarDir):
        shutil.rmtree(tarDir)
    os.makedirs(tarDir)
    for name in nameList:
        shutil.copyfile(os.path.join(rootDir, name), os.path.join(tarDir, name))


def get_source_data(name, data_dir, height, width, batch_size, workers):
    root = osp.join(data_dir, name)

    dataset = datasets.create(name, root, num_val=0.1)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    # use all training images on source dataset
    train_set = dataset.train
    num_classes = dataset.num_train_ids

    transformer = T.Compose([
        T.Resize((height, width)),
        T.ToTensor(),
        normalizer,
    ])

    extfeat_loader = DataLoader(
        Preprocessor(train_set, root=dataset.images_dir,
                     transform=transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return dataset, extfeat_loader


def calDis(qFeature, gFeature):  # 246s
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
    noiseFeat, pureFeat = feature[labels == -1, :], feature[labels != -1, :]
    labels = labels[labels != -1]
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
        co2Model = models.create(args.arch, num_classes=632, pretrained=False)
        co3Model = models.create(args.arch, num_classes=632, pretrained=False)
        co4Model = models.create(args.arch, num_classes=632, pretrained=False)
        co5Model = models.create(args.arch, num_classes=632, pretrained=False)
    elif args.src_dataset == 'market1501':
        model = models.create(args.arch, num_classes=676, pretrained=False)
        coModel = models.create(args.arch, num_classes=676, pretrained=False)
        co2Model = models.create(args.arch, num_classes=676, pretrained=False)
        co3Model = models.create(args.arch, num_classes=676, pretrained=False)
        co4Model = models.create(args.arch, num_classes=676, pretrained=False)
        co5Model = models.create(args.arch, num_classes=676, pretrained=False)
    elif args.src_dataset == 'msmt17':
        model = models.create(args.arch, num_classes=1041, pretrained=False)
        coModel = models.create(args.arch, num_classes=1041, pretrained=False)
        co2Model = models.create(args.arch, num_classes=1041, pretrained=False)
        co3Model = models.create(args.arch, num_classes=1041, pretrained=False)
        co4Model = models.create(args.arch, num_classes=1041, pretrained=False)
        co5Model = models.create(args.arch, num_classes=1041, pretrained=False)
    elif args.src_dataset == 'cuhk03':
        model = models.create(args.arch, num_classes=1230, pretrained=False)
        coModel = models.create(args.arch, num_classes=1230, pretrained=False)
        co2Model = models.create(args.arch, num_classes=1230, pretrained=False)
        co3Model = models.create(args.arch, num_classes=1230, pretrained=False)
        co4Model = models.create(args.arch, num_classes=1230, pretrained=False)
        co5Model = models.create(args.arch, num_classes=1230, pretrained=False)
    else:
        raise RuntimeError('Please specify the number of classes (ids) of the network.')

    # Load from checkpoint
    if args.resume:
        print('Resuming checkpoints from finetuned model on another dataset...\n')
        checkpoint = load_checkpoint(args.resume)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        coModel.load_state_dict(checkpoint['state_dict'], strict=False)
        co2Model.load_state_dict(checkpoint['state_dict'], strict=False)
        co3Model.load_state_dict(checkpoint['state_dict'], strict=False)
        co4Model.load_state_dict(checkpoint['state_dict'], strict=False)
        co5Model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        raise RuntimeWarning('Not using a pre-trained model.')
    model = nn.DataParallel(model).cuda()
    coModel = nn.DataParallel(coModel).cuda()
    co2Model = nn.DataParallel(co2Model).cuda()
    co3Model = nn.DataParallel(co3Model).cuda()
    co4Model = nn.DataParallel(co4Model).cuda()
    co5Model = nn.DataParallel(co5Model).cuda()

    evaluator = Evaluator(model, print_freq=args.print_freq)
    evaluator.evaluate(test_loader, tgt_dataset.query, tgt_dataset.gallery)
    # if args.evaluate: return

    # Criterion
    criterion = [
        TripletLoss(args.margin, args.num_instances, isAvg=False, use_semi=False).cuda(),
        TripletLoss(args.margin, args.num_instances, isAvg=False, use_semi=False).cuda(),
    ]

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr
    )
    coOptimizer = torch.optim.Adam(
        coModel.parameters(), lr=args.lr
    )
    
    co2Optimizer = torch.optim.Adam(
        co2Model.parameters(), lr = args.lr
    )
    
    co3Optimizer = torch.optim.Adam(
        co3Model.parameters(), lr = args.lr
    )
    
    co4Optimizer = torch.optim.Adam(
        co4Model.parameters(), lr = args.lr
    )

    co5Optimizer = torch.optim.Adam(
        co5Model.parameters(), lr=args.lr
    )

    optims = [optimizer, coOptimizer, co2Optimizer, co3Optimizer,co4Optimizer, co5Optimizer]

    # training stage transformer on input images
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transformer = T.Compose([
        T.Resize((args.height, args.width)),
        T.RandomHorizontalFlip(),
        T.ToTensor(), normalizer,
        T.RandomErasing(probability=0.5, sh=0.2, r1=0.3)
    ])
    
    data_transformer =  T.Compose([
        T.Resize((args.height,args.width)),
        T.ToTensor(),
        normalizer,
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
        print('Iteration {}: Extracting Target Dataset Features...'.format(iter_n + 1))
        target_features, _ = extract_features(model, tgt_extfeat_loader, print_freq=args.print_freq)
        # synchronization feature order with dataset.train
        target_features = torch.cat([target_features[f].unsqueeze(0) for f, _, _ in tgt_dataset.trainval], 0)
        # calculate distance and rerank result
        print('Calculating feature distances...')
        target_features = target_features.numpy()
        rerank_dist = re_ranking(source_features, target_features, lambda_value=args.lambda_value)
        if iter_n == 0:
            # DBSCAN cluster
            tri_mat = np.triu(rerank_dist, 1)  # tri_mat.dim=2
            tri_mat = tri_mat[np.nonzero(tri_mat)]  # tri_mat.dim=1
            tri_mat = np.sort(tri_mat, axis=None)
            top_num = np.round(args.rho * tri_mat.size).astype(int)
            eps = tri_mat[:top_num].mean()
            print('eps in cluster: {:.3f}'.format(eps))
            cluster = DBSCAN(eps=eps, min_samples=4, metric='precomputed', n_jobs=8)

        # select & cluster images as training set of this epochs
        print('Clustering and labeling...')
        labels = cluster.fit_predict(rerank_dist)
        num_ids = len(set(labels)) - 1
        print('Iteration {} have {} training ids'.format(iter_n + 1, num_ids))
        # generate new dataset
        new_dataset, unknown_dataset = [], []
        # assign label for target ones
        unknownLab = labelNoise(torch.from_numpy(target_features), torch.from_numpy(labels))
        # unknownFeats = target_features[labels==-1,:]
        unCounter, index = 0, 0
        from collections import defaultdict
        realIDs, fakeIDs = defaultdict(list), []
        for (fname, realPID, cam), label in zip(tgt_dataset.trainval, labels):
            if label == -1:
                unknown_dataset.append((fname, int(unknownLab[unCounter]), cam))  # unknown data
                fakeIDs.append(int(unknownLab[unCounter]))
                realIDs[realPID].append(index)
                unCounter += 1
                index += 1
                continue
            # dont need to change codes in trainer.py _parsing_input function and sampler function after add 0
            new_dataset.append((fname, label, cam))
            fakeIDs.append(label)
            realIDs[realPID].append(index)
            index += 1
        print('Iteration {} have {} training images'.format(iter_n + 1, len(new_dataset)))
        print('Iteration {} have {} outliers training images'.format(iter_n+1, len(unknown_dataset)))
        precision, recall, fscore = calScores(realIDs, np.asarray(fakeIDs))  # fakeIDs does not contain -1
        print('precision:{}, recall:{}, fscore: {}'.format(100 * precision, 100 * recall, fscore))

        train_loader = DataLoader(
            Preprocessor(new_dataset, root=tgt_dataset.images_dir, transform=train_transformer),
            batch_size=args.batch_size, num_workers=4,
            sampler=RandomIdentitySampler(new_dataset, args.num_instances),
            pin_memory=True, drop_last=True
        )
        # hard samples
        # noiseImgs = [name[1] for name in unknown_dataset]
        # saveAll(noiseImgs, tgt_dataset.images_dir, 'noiseImg')
        # import ipdb; ipdb.set_trace()
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
        
        #data_write_csv('tgt_dataset.trainval.csv',tgt_dataset.trainval)
        #data_write_csv('new_dataset.csv',new_dataset)

        train_all_loader = DataLoader(
        Preprocessor(new_dataset, root=tgt_dataset.images_dir,
                     transform=data_transformer),
        batch_size=args.batch_size, num_workers=4,
        shuffle=False, pin_memory=True)
        
        
        target_features2, _ = extract_features(model, train_all_loader, print_freq=args.print_freq)
        # synchronization feature order with dataset.train
        target_features2 = torch.cat([target_features2[f].unsqueeze(0) for f, _, _ in new_dataset], 0)
        # calculate distance and rerank result
        print('Calculating feature distances...')
        target_features2 = target_features2.numpy()
        rerank_dist2 = re_ranking(source_features, target_features2, lambda_value=args.lambda_value)
        if iter_n == 0:
            # DBSCAN cluster
            tri_mat2 = np.triu(rerank_dist2, 1)  # tri_mat2.dim=2
            tri_mat2 = tri_mat2[np.nonzero(tri_mat2)]  # tri_mat2.dim=1
            tri_mat2 = np.sort(tri_mat2, axis=None)
            top_num2 = np.round(args.rho * tri_mat2.size).astype(int)
            eps2 = tri_mat2[:top_num2].mean()
            print('eps2 in cluster: {:.3f}'.format(eps2))
            cluster2 = DBSCAN(eps=eps2, min_samples=4, metric='precomputed', n_jobs=8)

        # select & cluster images as training set of this epochs
        print('Clustering and labeling...')
        labels2 = cluster2.fit_predict(rerank_dist2)
        num_ids2 = len(set(labels2)) - 1
        print('Iteration {} have {} training ids'.format(iter_n + 1, num_ids2))
        # generate new dataset
        new_dataset2, unknown_dataset2 = [], []
        # assign label for target ones
        unknownLab2 = labelNoise(torch.from_numpy(target_features2), torch.from_numpy(labels2))
        # unknownFeats = target_features[labels==-1,:]
        unCounter, index = 0, 0
        from collections import defaultdict
        realIDs2, fakeIDs2 = defaultdict(list), []
        for (fname, realPID, cam), label in zip(new_dataset, labels2):
            if label == -1:
                unknown_dataset2.append((fname, int(unknownLab2[unCounter]), cam))  # unknown data
                fakeIDs2.append(int(unknownLab2[unCounter]))
                realIDs2[realPID].append(index)
                unCounter += 1
                index += 1
                continue
            # dont need to change codes in trainer.py _parsing_input function and sampler function after add 0
            new_dataset2.append((fname, label, cam))
            fakeIDs2.append(label)
            realIDs2[realPID].append(index)
            index += 1
        print('Iteration {} have {} inliers2 training images'.format(iter_n + 1, len(new_dataset2)))
        print('Iteration {} have {} outliers2 training images'.format(iter_n+1, len(unknown_dataset2)))
        precision2, recall2, fscore2 = calScores(realIDs2, np.asarray(fakeIDs2))  # fakeIDs2 does not contain -1
        print('precision2:{}, recall2:{}, fscore2: {}'.format(100 * precision2, 100 * recall2, fscore2))

        # train_inliers_loader = DataLoader(
            # Preprocessor(new_dataset2, root=tgt_dataset.images_dir, transform=train_transformer),
            # batch_size=args.batch_size, num_workers=4,
            # sampler=RandomIdentitySampler(new_dataset2, args.num_instances),
            # pin_memory=True, drop_last=True
        # )
        # hard samples
        # noiseImgs = [name[1] for name in unknown_dataset]
        # saveAll(noiseImgs, tgt_dataset.images_dir, 'noiseImg')
        # import ipdb; ipdb.set_trace()
        train_out2_loader = DataLoader(
            Preprocessor(unknown_dataset2, root=tgt_dataset.images_dir, transform=train_transformer),
            batch_size=args.batch_size, num_workers=4,
            sampler=RandomIdentitySampler(unknown_dataset2, args.num_instances),
            pin_memory=True, drop_last=True
        )
        
                #*********************************三次聚类**************************************
        print('***************second cluster*************')
        print('tgt_dataset.trainval type:{}'.format(type(tgt_dataset.trainval)))
        print('new_dataset2 type:{}'.format(type(new_dataset2)))
        #tgt_dataset.trainval type:<class 'list'>
        #new_dataset type:<class 'list'>
        
        #data_write_csv('tgt_dataset.trainval.csv',tgt_dataset.trainval)
        #data_write_csv('new_dataset.csv',new_dataset)

        train_in2_loader = DataLoader(
        Preprocessor(new_dataset2, root=tgt_dataset.images_dir,
                     transform=data_transformer),
        batch_size=args.batch_size, num_workers=4,
        shuffle=False, pin_memory=True)
        
        
        target_features3, _ = extract_features(model, train_in2_loader, print_freq=args.print_freq)
        # synchronization feature order with dataset.train
        target_features3 = torch.cat([target_features3[f].unsqueeze(0) for f, _, _ in new_dataset2], 0)
        # calculate distance and rerank result
        print('Calculating feature distances...')
        target_features3 = target_features3.numpy()
        rerank_dist3 = re_ranking(source_features, target_features3, lambda_value=args.lambda_value)
        if iter_n == 0:
            # DBSCAN cluster
            tri_mat3 = np.triu(rerank_dist3, 1)  # tri_mat2.dim=2
            tri_mat3 = tri_mat3[np.nonzero(tri_mat3)]  # tri_mat2.dim=1
            tri_mat3 = np.sort(tri_mat3, axis=None)
            top_num3 = np.round(args.rho * tri_mat3.size).astype(int)
            eps3 = tri_mat3[:top_num3].mean()
            print('eps3 in cluster: {:.3f}'.format(eps3))
            cluster3 = DBSCAN(eps=eps3, min_samples=4, metric='precomputed', n_jobs=8)

        # select & cluster images as training set of this epochs
        print('Clustering and labeling...')
        labels3 = cluster3.fit_predict(rerank_dist3)
        num_ids3 = len(set(labels3)) - 1
        print('Iteration {} have {} training ids'.format(iter_n + 1, num_ids3))
        # generate new dataset
        new_dataset3, unknown_dataset3 = [], []
        # assign label for target ones
        unknownLab3 = labelNoise(torch.from_numpy(target_features3), torch.from_numpy(labels3))
        # unknownFeats = target_features[labels==-1,:]
        unCounter, index = 0, 0
        from collections import defaultdict
        realIDs3, fakeIDs3 = defaultdict(list), []
        for (fname, realPID, cam), label in zip(new_dataset, labels3):
            if label == -1:
                unknown_dataset3.append((fname, int(unknownLab3[unCounter]), cam))  # unknown data
                fakeIDs3.append(int(unknownLab3[unCounter]))
                realIDs3[realPID].append(index)
                unCounter += 1
                index += 1
                continue
            # dont need to change codes in trainer.py _parsing_input function and sampler function after add 0
            new_dataset3.append((fname, label, cam))
            fakeIDs3.append(label)
            realIDs3[realPID].append(index)
            index += 1
        print('Iteration {} have {} inliers3 training images'.format(iter_n + 1, len(new_dataset3)))
        print('Iteration {} have {} outliers3 training images'.format(iter_n+1, len(unknown_dataset3)))
        precision3, recall3, fscore3 = calScores(realIDs3, np.asarray(fakeIDs3))  # fakeIDs3 does not contain -1
        print('precision3:{}, recall3:{}, fscore3: {}'.format(100 * precision3, 100 * recall3, fscore3))

        # train_in3_loader = DataLoader(
            # Preprocessor(new_dataset3, root=tgt_dataset.images_dir, transform=train_transformer),
            # batch_size=args.batch_size, num_workers=4,
            # sampler=RandomIdentitySampler(new_dataset3, args.num_instances),
            # pin_memory=True, drop_last=True
        # )
        # hard samples
        # noiseImgs = [name[1] for name in unknown_dataset]
        # saveAll(noiseImgs, tgt_dataset.images_dir, 'noiseImg')
        # import ipdb; ipdb.set_trace()
        train_out3_loader = DataLoader(
            Preprocessor(unknown_dataset3, root=tgt_dataset.images_dir, transform=train_transformer),
            batch_size=args.batch_size, num_workers=4,
            sampler=RandomIdentitySampler(unknown_dataset3, args.num_instances),
            pin_memory=True, drop_last=True
        )
        
        
                #*********************************四次聚类**************************************
        print('***************second cluster*************')
        print('tgt_dataset.trainval type:{}'.format(type(tgt_dataset.trainval)))
        print('new_dataset3 type:{}'.format(type(new_dataset3)))
        #tgt_dataset.trainval type:<class 'list'>
        #new_dataset type:<class 'list'>
        
        #data_write_csv('tgt_dataset.trainval.csv',tgt_dataset.trainval)
        #data_write_csv('new_dataset.csv',new_dataset)

        train_in3_loader = DataLoader(
        Preprocessor(new_dataset3, root=tgt_dataset.images_dir,
                     transform=data_transformer),
        batch_size=args.batch_size, num_workers=4,
        shuffle=False, pin_memory=True)
        
        
        target_features4, _ = extract_features(model, train_in3_loader, print_freq=args.print_freq)
        # synchronization feature order with dataset.train
        target_features4 = torch.cat([target_features4[f].unsqueeze(0) for f, _, _ in new_dataset3], 0)
        # calculate distance and rerank result
        print('Calculating feature distances...')
        target_features4 = target_features4.numpy()
        rerank_dist4 = re_ranking(source_features, target_features4, lambda_value=args.lambda_value)
        if iter_n == 0:
            # DBSCAN4cluster
            tri_mat4 = np.triu(rerank_dist4, 1)  # tri_mat4.dim=2
            tri_mat4 = tri_mat4[np.nonzero(tri_mat4)]  # tri_mat4.dim=1
            tri_mat4 = np.sort(tri_mat4, axis=None)
            top_num4 = np.round(args.rho * tri_mat4.size).astype(int)
            eps4 = tri_mat4[:top_num4].mean()
            print('eps4 in cluster: {:.3f}'.format(eps4))
            cluster4 = DBSCAN(eps=eps4, min_samples=4, metric='precomputed', n_jobs=8)

        # select & cluster images as training set of this epochs
        print('Clustering and labeling...')
        labels4 = cluster4.fit_predict(rerank_dist4)
        num_ids4 = len(set(labels4)) - 1
        print('Iteration {} have {} training ids'.format(iter_n + 1, num_ids4))
        # generate new dataset
        new_dataset4, unknown_dataset4 = [], []
        # assign label for target ones
        unknownLab4 = labelNoise(torch.from_numpy(target_features4), torch.from_numpy(labels4))
        # unknownFeats = target_features[labels==-1,:]
        unCounter, index = 0, 0
        from collections import defaultdict
        realIDs4, fakeIDs4 = defaultdict(list), []
        for (fname, realPID, cam), label in zip(new_dataset, labels4):
            if label == -1:
                unknown_dataset4.append((fname, int(unknownLab4[unCounter]), cam))  # unknown data
                fakeIDs4.append(int(unknownLab4[unCounter]))
                realIDs4[realPID].append(index)
                unCounter += 1
                index += 1
                continue
            # dont need to change codes in trainer.py _parsing_input function and sampler function after add 0
            new_dataset4.append((fname, label, cam))
            fakeIDs4.append(label)
            realIDs4[realPID].append(index)
            index += 1
        print('Iteration {} have {} inliers4 training images'.format(iter_n + 1, len(new_dataset4)))
        print('Iteration {} have {} outliers4 training images'.format(iter_n+1, len(unknown_dataset4)))
        precision4, recall4, fscore4 = calScores(realIDs4, np.asarray(fakeIDs4))  # fakeIDs4 does not contain -1
        print('precision4:{}, recall4:{}, fscore4: {}'.format(100 * precision4, 100 * recall4, fscore4))

        # train_in4_loader = DataLoader(
        #     Preprocessor(new_dataset4, root=tgt_dataset.images_dir, transform=train_transformer),
        #     batch_size=args.batch_size, num_workers=4,
        #     sampler=RandomIdentitySampler(new_dataset4, args.num_instances),
        #     pin_memory=True, drop_last=True
        # )
        # hard samples
        # noiseImgs = [name[1] for name in unknown_dataset]
        # saveAll(noiseImgs, tgt_dataset.images_dir, 'noiseImg')
        # import ipdb; ipdb.set_trace()
        train_out4_loader = DataLoader(
            Preprocessor(unknown_dataset4, root=tgt_dataset.images_dir, transform=train_transformer),
            batch_size=args.batch_size, num_workers=4,
            sampler=RandomIdentitySampler(unknown_dataset4, args.num_instances),
            pin_memory=True, drop_last=True
        )

        # *********************************五次聚类**************************************
        print('***************second cluster*************')
        print('tgt_dataset.trainval type:{}'.format(type(tgt_dataset.trainval)))
        print('new_dataset4 type:{}'.format(type(new_dataset4)))
        # tgt_dataset.trainval type:<class 'list'>
        # new_dataset type:<class 'list'>

        # data_write_csv('tgt_dataset.trainval.csv',tgt_dataset.trainval)
        # data_write_csv('new_dataset.csv',new_dataset)

        train_in4_loader = DataLoader(
            Preprocessor(new_dataset4, root=tgt_dataset.images_dir,
                         transform=data_transformer),
            batch_size=args.batch_size, num_workers=4,
            shuffle=False, pin_memory=True)

        target_features5, _ = extract_features(model, train_in4_loader, print_freq=args.print_freq)
        # synchronization feature order with dataset.train
        target_features5 = torch.cat([target_features5[f].unsqueeze(0) for f, _, _ in new_dataset4], 0)
        # calculate distance and rerank result
        print('Calculating feature distances...')
        target_features5 = target_features5.numpy()
        rerank_dist5 = re_ranking(source_features, target_features5, lambda_value=args.lambda_value)
        if iter_n == 0:
            # DBSCAN4cluster
            tri_mat5 = np.triu(rerank_dist5, 1)  # tri_mat4.dim=2
            tri_mat5 = tri_mat5[np.nonzero(tri_mat5)]  # tri_mat4.dim=1
            tri_mat5 = np.sort(tri_mat5, axis=None)
            top_num5 = np.round(args.rho * tri_mat5.size).astype(int)
            eps5 = tri_mat5[:top_num5].mean()
            print('eps4 in cluster: {:.3f}'.format(eps5))
            cluster5 = DBSCAN(eps=eps5, min_samples=4, metric='precomputed', n_jobs=8)

        # select & cluster images as training set of this epochs
        print('Clustering and labeling...')
        labels5 = cluster5.fit_predict(rerank_dist5)
        num_ids5 = len(set(labels5)) - 1
        print('Iteration {} have {} training ids'.format(iter_n + 1, num_ids5))
        # generate new dataset
        new_dataset5, unknown_dataset5 = [], []
        # assign label for target ones
        unknownLab5 = labelNoise(torch.from_numpy(target_features5), torch.from_numpy(labels5))
        # unknownFeats = target_features[labels==-1,:]
        unCounter, index = 0, 0
        from collections import defaultdict
        realIDs5, fakeIDs5 = defaultdict(list), []
        for (fname, realPID, cam), label in zip(new_dataset, labels5):
            if label == -1:
                unknown_dataset5.append((fname, int(unknownLab5[unCounter]), cam))  # unknown data
                fakeIDs5.append(int(unknownLab5[unCounter]))
                realIDs5[realPID].append(index)
                unCounter += 1
                index += 1
                continue
            # dont need to change codes in trainer.py _parsing_input function and sampler function after add 0
            new_dataset5.append((fname, label, cam))
            fakeIDs5.append(label)
            realIDs5[realPID].append(index)
            index += 1
        print('Iteration {} have {} inliers5 training images'.format(iter_n + 1, len(new_dataset5)))
        print('Iteration {} have {} outliers5 training images'.format(iter_n + 1, len(unknown_dataset5)))
        precision5, recall5, fscore5 = calScores(realIDs5, np.asarray(fakeIDs5))  # fakeIDs5 does not contain -1
        print('precision5:{}, recall5:{}, fscore5: {}'.format(100 * precision5, 100 * recall5, fscore5))

        train_in5_loader = DataLoader(
            Preprocessor(new_dataset5, root=tgt_dataset.images_dir, transform=train_transformer),
            batch_size=args.batch_size, num_workers=4,
            sampler=RandomIdentitySampler(new_dataset5, args.num_instances),
            pin_memory=True, drop_last=True
        )
        # hard samples
        # noiseImgs = [name[1] for name in unknown_dataset]
        # saveAll(noiseImgs, tgt_dataset.images_dir, 'noiseImg')
        # import ipdb; ipdb.set_trace()
        train_out5_loader = DataLoader(
            Preprocessor(unknown_dataset5, root=tgt_dataset.images_dir, transform=train_transformer),
            batch_size=args.batch_size, num_workers=4,
            sampler=RandomIdentitySampler(unknown_dataset5, args.num_instances),
            pin_memory=True, drop_last=True
        )



        # train model with new generated dataset
        trainer = CoTrainerAsy6(
            model, coModel, co2Model, co3Model, co4Model, co5Model, train_in5_loader, train_out5_loader, train_out4_loader, train_out3_loader, train_out2_loader, unLoader, criterion, optims
        )
        
        # trainer = CoTrainerAsy(
            # model, coModel, train_loader, unLoader, criterion, optims
        # )

        # Start training
        for epoch in range(args.epochs):
            trainer.train(epoch, remRate=0.2 + (0.8 / args.iteration) * (1 + iter_n))

        # test only
        rank_score = evaluator.evaluate(test_loader, tgt_dataset.query, tgt_dataset.gallery)
        # print('co-model:\n')
        # rank_score = evaluatorB.evaluate(test_loader, tgt_dataset.query, tgt_dataset.gallery)

    # Evaluate
    rank_score = evaluator.evaluate(test_loader, tgt_dataset.query, tgt_dataset.gallery)
    save_checkpoint({
        'state_dict': model.module.state_dict(),
        'epoch': epoch + 1, 'best_top1': rank_score.market1501[0],
    }, True, fpath=osp.join(args.logs_dir, 'asyCo.pth'))
    return rank_score.map, rank_score.market1501[0]


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
                        default='')
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
