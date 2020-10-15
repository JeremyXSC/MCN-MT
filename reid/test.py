#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from __future__ import print_function, absolute_import
import argparse
import time
import os.path as osp
import os 
import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.backends import cudnn
from torch.utils.data import DataLoader
from sklearn import metrics
import scipy.sparse as sp
import warnings

import numpy as np
from sklearn import mixture
from sklearn.mixture import GaussianMixture
print('Clustering and labeling...')

#X = np.array([[1, 2], [1, 4], [1, 0],[4, 2], [4, 4], [4, 0]])
X = np.array([[1.22222, 2], [1.222, 2], [1.222, 2],[4, 4.3333], [4, 4.333], [4, 4.33]])

#X = np.genfromtxt('../FINCH-Clustering-master/data/STL-10/data.txt', delimiter=",").astype(np.float32)
 
#print(X,'********',X1)
#print(X.size)
print(X.shape)
#X = X.reshape((13000,2048))
#print(X.shape)
 
'''g = GaussianMixture(n_components=200,n_init=1,max_iter=10,tol=1e-2,reg_covar=5e-4)
g.fit(X)
labels = g.predict(X)
print('labels are {}'.format(labels))
centers = g.means_
print('centers are {}'.format(centers))'''

g = GaussianMixture(n_components=2,n_init=1,max_iter=10,tol=1e-2,reg_covar=5e-4)
RES = g.fit(X)
labels = RES.predict(X)
print('labels are {}'.format(labels))
centers = RES.means_
print('centers are {}'.format(centers))


'''prob = g.predict_proba(X)     #或者还是用后验概率每一列最大的那个值
print('prob are {}'.format(prob))
m = g.means_     #是否可以直接用means_当作center
print('m are {}'.format(m))
mm = g.means_.argmax()
print('mm are {}'.format(mm))
prob1 = prob[:,g.means_.argmax()]
print('prob1 are {}'.format(prob1))'''