from __future__ import print_function, absolute_import
import time

import torch
from torch.autograd import Variable

from .utils.meters import AverageMeter
import numpy as np


class BaseTrainer(object):
    def __init__(self, model, criterions, print_freq=1):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterions = criterions
        self.print_freq = print_freq

    def train(self, epoch, data_loader, optimizer):
        self.model.train()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()

        end = time.time()
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)
            inputs, targets = self._parse_data(inputs)  # image and pid
            loss, prec1 = self._forward(inputs, targets, epoch)
            losses.update(loss.item(), targets.size(0))
            precisions.update(prec1, targets.size(0))
            optimizer.zero_grad()
            loss.backward()
            # add gradient clip for lstm
            for param in self.model.parameters():
                try:
                    param.grad.data.clamp(-1., 1.)
                except:
                    continue
            optimizer.step()
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % self.print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              precisions.val, precisions.avg))

    def _parse_data(self, inputs):
        raise NotImplementedError

    def _forward(self, inputs, targets):
        raise NotImplementedError


class CoTeaching(object):
    def __init__(self, model, coModel, newDataSet, criterions, optimizers, print_freq=1):
        self.modelA = model
        self.modelB = coModel  # the co-teacher
        self.newDataSet = newDataSet
        self.criterions = criterions
        self.optimizers = optimizers
        self.print_freq = print_freq

    def train(self, epoch, remRate=0.2):
        self.modelA.train()
        self.modelB.train()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()
        end = time.time()

        for i, inputs in enumerate(self.newDataSet):
            data_time.update(time.time() - end)
            inputs, targets = self._parse_data(inputs)  # image and pid
            if i % 2 == 0:
                # update CNNB
                lossPure, prec1 = self._forward(inputs, targets, epoch, self.modelA)  # assigned samples
                # noise sample mining
                lossIdx = np.argsort(lossPure.data.cpu()).cuda()
                inputsCNNB, tarCNNB = inputs[0][lossIdx], targets[lossIdx]
                inputsCNNB, tarCNNB = [inputsCNNB[:int(remRate * lossIdx.shape[0]), ...]], tarCNNB[:int(
                    remRate * lossIdx.shape[0])]
                # loss for cnn B
                lossCNNB, precCNNB = self._forward(inputsCNNB, tarCNNB, epoch, self.modelB)
                lossCNNB = lossCNNB.mean()
                losses.update(lossCNNB.item(), tarCNNB.size(0))
                precisions.update(precCNNB, tarCNNB.size(0))
                self.optimizers[1].zero_grad()
                lossCNNB.backward()
                for param in self.modelB.parameters():
                    try:
                        param.grad.data.clamp(-1., 1.)
                    except:
                        continue
                self.optimizers[1].step()
            else:
                # update CNNA
                lossPure, prec1 = self._forward(inputs, targets, epoch, self.modelB)
                # sample mining
                lossIdx = np.argsort(lossPure.data.cpu()).cuda()
                inputsCNNA, tarCNNA = inputs[0][lossIdx], targets[lossIdx]
                inputsCNNA, tarCNNA = [inputsCNNA[:int(remRate * lossIdx.shape[0]), ...]], tarCNNA[:int(
                    remRate * lossIdx.shape[0])]
                # pure noise loss
                lossCNNA, precCNNA = self._forward(inputsCNNA, tarCNNA, epoch, self.modelA)
                lossCNNA = lossCNNA.mean()
                # update
                losses.update(lossCNNA.item(), tarCNNA.size(0))
                precisions.update(precCNNA, tarCNNA.size(0))
                # update CNNA
                self.optimizers[0].zero_grad()
                lossCNNA.backward()
                for param in self.modelA.parameters():
                    try:
                        param.grad.data.clamp(-1., 1.)
                    except:
                        continue
                # update modelA
                self.optimizers[0].step()

            batch_time.update(time.time() - end)
            end = time.time()
            if (i + 1) % self.print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(self.newDataSet),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              precisions.val, precisions.avg))

    def _parse_data(self, inputs):
        imgs, fname, pids, _ = inputs
        inputs = [Variable(imgs).cuda()]
        targets = Variable(pids.cuda())
        return inputs, targets  # image and pid

    def _forward(self, inputs, targets, epoch, model):
        outputs = model(*inputs)  # outputs=[x1,x2,x3]
        # x1 triplet loss
        loss_tri, _ = self.criterions[0](outputs[0], targets, epoch)  # feature
        # x2 triplet loss
        loss_global, prec_global = self.criterions[1](outputs[1], targets, epoch)
        return loss_tri + loss_global, prec_global


class Trainer(BaseTrainer):
    def _parse_data(self, inputs):
        imgs, _, pids, _ = inputs
        inputs = [Variable(imgs)]
        targets = Variable(pids.cuda())
        return inputs, targets  # image and pid

    def _forward(self, inputs, targets, epoch):
        outputs = self.model(*inputs)  # outputs=[x1,x2,x3]
        # new added by wc
        # x1 triplet loss
        loss_tri, prec_tri = self.criterions[0](outputs[0], targets, epoch)  # feature
        # x2 triplet loss
        loss_global, prec_global = self.criterions[1](outputs[1], targets, epoch)  # fc
        # loss_center = self.criterions[2](outputs[0], targets)
        return loss_tri + loss_global, prec_global


class RCoTeaching(object):
    """
    RCT implemention
    """

    def __init__(self, model, coModel, newDataSet, noiseDataSet, criterions, optimizers, print_freq=1):
        self.modelA = model
        self.modelB = coModel  # the co-teacher
        self.noiseData = noiseDataSet
        self.newDataSet = newDataSet
        self.criterions = criterions
        self.optimizers = optimizers
        self.print_freq = print_freq

    def train(self, epoch, remRate=0.2):
        self.modelA.train()
        self.modelB.train()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()
        end = time.time()

        for i, inputs in enumerate(self.newDataSet):
            data_time.update(time.time() - end)
            inputs, targets = self._parse_data(inputs)  # image and pid
            # noise data
            try:
                noiseInput = next(self.noiseData)
            except:
                noiseLoader = iter(self.noiseData)
                noiseInput = next(noiseLoader)
            noiseInput, noiseLab = self._parse_data(noiseInput)
            if i % 2 != 0:
                # update CNNA
                lossNoise, _ = self._forward(noiseInput, noiseLab, epoch, self.modelB)  # assigned samples
                lossPure, _ = self._forward(inputs, targets, epoch, self.modelB)
                # # assigned's easy samples
                lossIdx, lossPureIdx = np.argsort(lossNoise.data.cpu()).cuda(), np.argsort(lossPure.data).cuda()
                smallNoise = noiseInput[0][lossIdx[:int(remRate * lossNoise.shape[0])], ...]
                smallPure = inputs[0][lossPureIdx[:int(remRate * lossPure.shape[0])], ...]
                smallNoiseLab = noiseLab[lossIdx[:int(remRate * lossNoise.shape[0])]]
                smallPureLab = targets[lossPureIdx[:int(remRate * lossPure.shape[0])]]
                newLab = torch.cat([smallNoiseLab, smallPureLab])
                lossCNNA, precCNNA = self._forward([torch.cat([smallNoise, smallPure])], newLab, epoch, self.modelA)
                lossCNNA = lossCNNA.mean()
                losses.update(lossCNNA.item(), newLab.size(0))
                precisions.update(precCNNA, newLab.size(0))
                self.optimizers[0].zero_grad()
                lossCNNA.backward()
                for param in self.modelA.parameters():
                    try:
                        param.grad.data.clamp(-1., 1.)
                    except:
                        continue
                self.optimizers[0].step()
            else:
                # update CNNB
                lossNoise, _ = self._forward(noiseInput, noiseLab, epoch, self.modelA)  # assigned samples
                lossPure, _ = self._forward(inputs, targets, epoch, self.modelA)
                # # assigned's easy samples
                lossIdx, lossPureIdx = np.argsort(lossNoise.data.cpu()).cuda(), np.argsort(lossPure.data.cpu()).cuda()
                smallNoise = noiseInput[0][lossIdx[:int(remRate * lossNoise.shape[0])], ...]
                smallPure = inputs[0][lossPureIdx[:int(remRate * lossPure.shape[0])], ...]
                smallNoiseLab = noiseLab[lossIdx[:int(remRate * lossNoise.shape[0])]]
                smallPureLab = targets[lossPureIdx[:int(remRate * lossPure.shape[0])]]
                newLab = torch.cat([smallNoiseLab, smallPureLab])
                lossCNNB, precCNNB = self._forward([torch.cat([smallNoise, smallPure])], newLab, epoch, self.modelB)
                lossCNNB = lossCNNB.mean()
                losses.update(lossCNNB.item(), newLab.size(0))
                precisions.update(precCNNB, newLab.size(0))
                self.optimizers[1].zero_grad()
                lossCNNB.backward()
                for param in self.modelB.parameters():
                    try:
                        param.grad.data.clamp(-1., 1.)
                    except:
                        continue
                self.optimizers[1].step()

            batch_time.update(time.time() - end)
            end = time.time()
            if (i + 1) % self.print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(self.newDataSet),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              precisions.val, precisions.avg))

    def _parse_data(self, inputs):
        imgs, fname, pids, _ = inputs
        inputs = [Variable(imgs).cuda()]
        targets = Variable(pids.cuda())
        return inputs, targets  # image and pid

    def _forward(self, inputs, targets, epoch, model):
        outputs = model(*inputs)  # outputs=[x1,x2,x3]
        # x1 triplet loss
        loss_tri, _ = self.criterions[0](outputs[0], targets, epoch)  # feature
        # x2 triplet loss
        loss_global, prec_global = self.criterions[1](outputs[1], targets, epoch)
        return loss_tri + loss_global, prec_global


class CoTrainerAsy(object):
    def saveImg(self, imgList, gtList):
        import shutil
        import os
        rootDir = self.noiseData.dataset.root
        if os.path.exists('smallLoss'):
            shutil.rmtree('smallLoss')
        os.makedirs('smallLoss')
        for name, pid in zip(imgList, gtList):
            curPath = os.path.join(rootDir, name)
            nameList = name.split('_')
            nameList[0] = str(pid)
            tarPath = os.path.join('smallLoss', '_'.join(nameList))
            shutil.copyfile(curPath, tarPath)

    def __init__(self, model, coModel, newDataSet, noiseDataSet, criterions, optimizers, print_freq=1):
        self.modelA = model
        self.modelB = coModel  # the co-teacher
        self.noiseData = noiseDataSet
        self.newDataSet = newDataSet
        self.criterions = criterions
        self.optimizers = optimizers
        self.print_freq = print_freq

    def train(self, epoch, remRate=0.2):
        self.modelA.train()
        self.modelB.train()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()
        end = time.time()

        for i, inputs in enumerate(self.newDataSet):
            data_time.update(time.time() - end)
            inputs, targets, names = self._parse_data(inputs)  # image and pid
            if i % 2 == 0:
                # update CNNB
                lossPure, prec1 = self._forward(inputs, targets, epoch, self.modelA)  # assigned samples
                # # assigned's easy samples
                lossIdx = np.argsort(lossPure.data.cpu()).cuda()
                pureInput = [inputs[0][lossIdx[:int(remRate * lossPure.shape[0])], ...]]
                pureLab = targets[lossIdx[:int(remRate * lossPure.shape[0])]].long()
                # loss for cnn B
                lossCNNB, precCNNB = self._forward(pureInput, pureLab, epoch, self.modelB)
                lossCNNB = lossCNNB.mean()
                losses.update(lossCNNB.item(), pureLab.size(0))
                precisions.update(precCNNB, pureLab.size(0))
                self.optimizers[1].zero_grad()
                lossCNNB.backward()
                for param in self.modelB.parameters():
                    try:
                        param.grad.data.clamp(-1., 1.)
                    except:
                        continue
                self.optimizers[1].step()
            else:
                # update CNNA
                try:
                    noiseInput = next(self.noiseData)
                except:
                    noiseLoader = iter(self.noiseData)
                    noiseInput = next(noiseLoader)
                noiseInput, noiseLab, noiseNames = self._parse_data(noiseInput)
                lossNoise, prec1 = self._forward(noiseInput, noiseLab, epoch, self.modelB)
                # sample mining
                lossIdx = np.argsort(lossNoise.data.cpu()).cuda()
                noiseInput, noiseLab = noiseInput[0][lossIdx], noiseLab[lossIdx]
                noiseInput, noiseLab = noiseInput[:int(remRate * lossNoise.shape[0]), ...], noiseLab[:int(
                    remRate * lossNoise.shape[0])]
                # save small samples
                # noiseImg = np.asarray(noiseNames)[lossIdx][:int(remRate*lossNoise.shape[0])]
                # self.saveImg(noiseImg, noiseLab) # save image
                # mix update, part assigned and part unassigned
                mixInput, mixLab = [torch.cat([inputs[0], noiseInput])], torch.cat([targets.long(), noiseLab])
                lossMix, precCNNA = self._forward(mixInput, mixLab, epoch, self.modelA)
                lossCNNA = lossMix.mean()
                # update
                losses.update(lossCNNA.item(), mixLab.size(0))
                precisions.update(precCNNA, mixLab.size(0))
                # update CNNA
                self.optimizers[0].zero_grad()
                lossCNNA.backward()
                for param in self.modelA.parameters():
                    try:
                        param.grad.data.clamp(-1., 1.)
                    except:
                        continue
                # update modelA
                self.optimizers[0].step()

            batch_time.update(time.time() - end)
            end = time.time()
            if (i + 1) % self.print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(self.newDataSet),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              precisions.val, precisions.avg))

    def _parse_data(self, inputs):
        imgs, fname, pids, _ = inputs
        inputs = [Variable(imgs).cuda()]
        targets = Variable(pids.cuda())
        return inputs, targets, fname  # image and pid

    def _forward(self, inputs, targets, epoch, model):
        outputs = model(*inputs)  # outputs=[x1,x2,x3]
        # x1 triplet loss
        loss_tri, _ = self.criterions[0](outputs[0], targets, epoch)  # feature
        # x2 triplet loss
        loss_global, prec_global = self.criterions[1](outputs[1], targets, epoch)
        return loss_tri + loss_global, prec_global

# trainer = CoTrainerAsyMean(
            # model, coModel, model_ema, coModel_ema, train_loader, unLoader, criterion, optims, alpha=args.alpha
        # )
class CoTrainerAsyMean(object):
    def saveImg(self, imgList, gtList):
        import shutil
        import os
        rootDir = self.noiseData.dataset.root
        if os.path.exists('smallLoss'):
            shutil.rmtree('smallLoss')
        os.makedirs('smallLoss')
        for name, pid in zip(imgList, gtList):
            curPath = os.path.join(rootDir, name)
            nameList = name.split('_')
            nameList[0] = str(pid)
            tarPath = os.path.join('smallLoss', '_'.join(nameList))
            shutil.copyfile(curPath, tarPath)

    def __init__(self, model, coModel, model_ema, coModel_ema, newDataSet, noiseDataSet, criterions, optimizers, alpha=0.9, print_freq=1):
        self.modelA = model
        self.modelB = coModel  # the co-teacher
        self.modelA_ema = model_ema
        self.modelB_ema = coModel_ema
        self.noiseData = noiseDataSet
        self.newDataSet = newDataSet
        self.criterions = criterions
        self.optimizers = optimizers
        self.alpha = alpha
        self.print_freq = print_freq

    def train(self, epoch, remRate=0.2):
        self.modelA.train()
        self.modelB.train()
        self.modelA_ema.train()
        self.modelB_ema.train()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()
        #precisions = [AverageMeter(),AverageMeter()]
        end = time.time()

        for i, inputs in enumerate(self.newDataSet):
            data_time.update(time.time() - end)
            inputs, targets, names = self._parse_data(inputs)  # image and pid
            if i % 2 == 0:
                # update CNNB
                lossPure, prec1 = self._forward(inputs, targets, epoch, self.modelA)  # assigned samples
                #lossPure_ema, prec1_ema = self._forward(inputs, targets, epoch, self.modelA_ema)
                # # assigned's easy samples
                lossIdx = np.argsort(lossPure.data.cpu()).cuda()
                #lossIdx_ema = np.argsort(lossPure_ema.data.cpu()).cuda()
                pureInput = [inputs[0][lossIdx[:int(remRate * lossPure.shape[0])], ...]]
                #pureInput_ema = [inputs[0][lossIdx_ema[:int(remRate * lossPure_ema.shape[0])], ...]]
                pureLab = targets[lossIdx[:int(remRate * lossPure.shape[0])]].long()
                #pureLab_ema = targets[lossIdx_ema[:int(remRate * lossPure_ema.shape[0])]].long()
                # loss for cnn B
                lossCNNB, precCNNB = self._forward(pureInput, pureLab, epoch, self.modelB)
                lossCNNB_ema, precCNNB_ema = self._forward(pureInput, pureLab, epoch, self.modelB_ema)
                lossCNNB = lossCNNB.mean()
                lossCNNB_ema = lossCNNB_ema.mean()
                lossCNNB_mean = (lossCNNB + lossCNNB_ema)*0.5
                losses.update(lossCNNB_mean.item(), pureLab.size(0))
                precisions.update(precCNNB, pureLab.size(0))
                self.optimizers[1].zero_grad()
                lossCNNB_mean.backward()
                for param in self.modelB.parameters():#梯度截断
                    try:
                        param.grad.data.clamp(-1., 1.)
                    except:
                        continue
                self.optimizers[1].step()
                
                self._update_ema_variables(self.modelB, self.modelB_ema, self.alpha, epoch*len(self.newDataSet)+i)
                for param in self.modelB_ema.parameters():#梯度截断
                    try:
                        param.grad.data.clamp(-1., 1.)
                    except:
                        continue
                
            else:
                # update CNNA
                try:
                    noiseInput = next(self.noiseData)
                except:
                    noiseLoader = iter(self.noiseData)
                    noiseInput = next(noiseLoader)
                noiseInput, noiseLab, noiseNames = self._parse_data(noiseInput)
                lossNoise, prec1 = self._forward(noiseInput, noiseLab, epoch, self.modelB)
                #lossNoise_ema, prec1_ema = self._forward(noiseInput, noiseLab, epoch, self.modelB_ema)
                # sample mining
                lossIdx = np.argsort(lossNoise.data.cpu()).cuda()
                noiseInput, noiseLab = noiseInput[0][lossIdx], noiseLab[lossIdx]
                noiseInput, noiseLab = noiseInput[:int(remRate * lossNoise.shape[0]), ...], noiseLab[:int(
                    remRate * lossNoise.shape[0])]
                # save small samples
                # noiseImg = np.asarray(noiseNames)[lossIdx][:int(remRate*lossNoise.shape[0])]
                # self.saveImg(noiseImg, noiseLab) # save image
                # mix update, part assigned and part unassigned
                mixInput, mixLab = [torch.cat([inputs[0], noiseInput])], torch.cat([targets.long(), noiseLab])
                lossMix, precCNNA = self._forward(mixInput, mixLab, epoch, self.modelA)
                lossMix_ema, precCNNA_ema = self._forward(mixInput, mixLab, epoch, self.modelA_ema)
                lossCNNA = lossMix.mean()
                lossCNNA_ema = lossMix_ema.mean()
                lossCNNA_mean = (lossCNNA + lossCNNA_ema)*0.5
                # update
                losses.update(lossCNNA_mean.item(), mixLab.size(0))
                precisions.update(precCNNA, mixLab.size(0))
                # update CNNA
                self.optimizers[0].zero_grad()
                lossCNNA_mean.backward()
                for param in self.modelA.parameters():
                    try:
                        param.grad.data.clamp(-1., 1.)
                    except:
                        continue
                # update modelA
                self.optimizers[0].step()
                
                self._update_ema_variables(self.modelA, self.modelA_ema, self.alpha, epoch*len(self.newDataSet)+i)
                for param in self.modelA_ema.parameters():
                    try:
                        param.grad.data.clamp(-1., 1.)
                    except:
                        continue

            batch_time.update(time.time() - end)
            end = time.time()
            if (i + 1) % self.print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(self.newDataSet),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              precisions.val, precisions.avg))
    
    def _update_ema_variables(self, model, ema_model, alpha, global_step):
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
        
    def _parse_data(self, inputs):
        imgs, fname, pids, _ = inputs
        inputs = [Variable(imgs).cuda()]
        targets = Variable(pids.cuda())
        return inputs, targets, fname  # image and pid

    def _forward(self, inputs, targets, epoch, model):
        outputs = model(*inputs)  # outputs=[x1,x2,x3]
        # x1 triplet loss
        loss_tri, _ = self.criterions[0](outputs[0], targets, epoch)  # feature
        # x2 triplet loss
        loss_global, prec_global = self.criterions[1](outputs[1], targets, epoch)
        return loss_tri + loss_global, prec_global

class CoTrainerAsyMean1(object):#用ema assign samples
    def saveImg(self, imgList, gtList):
        import shutil
        import os
        rootDir = self.noiseData.dataset.root
        if os.path.exists('smallLoss'):
            shutil.rmtree('smallLoss')
        os.makedirs('smallLoss')
        for name, pid in zip(imgList, gtList):
            curPath = os.path.join(rootDir, name)
            nameList = name.split('_')
            nameList[0] = str(pid)
            tarPath = os.path.join('smallLoss', '_'.join(nameList))
            shutil.copyfile(curPath, tarPath)

    def __init__(self, model, coModel, model_ema, coModel_ema, newDataSet, noiseDataSet, criterions, optimizers, alpha=0.9, print_freq=1):
        self.modelA = model
        self.modelB = coModel  # the co-teacher
        self.modelA_ema = model_ema
        self.modelB_ema = coModel_ema
        self.noiseData = noiseDataSet
        self.newDataSet = newDataSet
        self.criterions = criterions
        self.optimizers = optimizers
        self.alpha = alpha
        self.print_freq = print_freq

    def train(self, epoch, remRate=0.2):
        self.modelA.train()
        self.modelB.train()
        self.modelA_ema.train()
        self.modelB_ema.train()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()
        #precisions = [AverageMeter(),AverageMeter()]
        end = time.time()

        for i, inputs in enumerate(self.newDataSet):
            data_time.update(time.time() - end)
            inputs, targets, names = self._parse_data(inputs)  # image and pid
            if i % 2 == 0:
                # update CNNB
                lossPure, prec1 = self._forward(inputs, targets, epoch, self.modelA_ema)  # assigned samples
                #lossPure_ema, prec1_ema = self._forward(inputs, targets, epoch, self.modelA_ema)
                # # assigned's easy samples
                lossIdx = np.argsort(lossPure.data.cpu()).cuda()
                #lossIdx_ema = np.argsort(lossPure_ema.data.cpu()).cuda()
                pureInput = [inputs[0][lossIdx[:int(remRate * lossPure.shape[0])], ...]]
                #pureInput_ema = [inputs[0][lossIdx_ema[:int(remRate * lossPure_ema.shape[0])], ...]]
                pureLab = targets[lossIdx[:int(remRate * lossPure.shape[0])]].long()
                #pureLab_ema = targets[lossIdx_ema[:int(remRate * lossPure_ema.shape[0])]].long()
                # loss for cnn B
                lossCNNB, precCNNB = self._forward(pureInput, pureLab, epoch, self.modelB)
                lossCNNB_ema, precCNNB_ema = self._forward(pureInput, pureLab, epoch, self.modelB_ema)
                lossCNNB = lossCNNB.mean()
                lossCNNB_ema = lossCNNB_ema.mean()
                lossCNNB_mean = (lossCNNB + lossCNNB_ema)*0.5
                losses.update(lossCNNB_mean.item(), pureLab.size(0))
                precisions.update(precCNNB, pureLab.size(0))
                self.optimizers[1].zero_grad()
                lossCNNB_mean.backward()
                for param in self.modelB.parameters():#梯度截断
                    try:
                        param.grad.data.clamp(-1., 1.)
                    except:
                        continue
                self.optimizers[1].step()
                
                self._update_ema_variables(self.modelB, self.modelB_ema, self.alpha, epoch*len(self.newDataSet)+i)
                for param in self.modelB_ema.parameters():#梯度截断
                    try:
                        param.grad.data.clamp(-1., 1.)
                    except:
                        continue
                
            else:
                # update CNNA
                try:
                    noiseInput = next(self.noiseData)
                except:
                    noiseLoader = iter(self.noiseData)
                    noiseInput = next(noiseLoader)
                noiseInput, noiseLab, noiseNames = self._parse_data(noiseInput)
                lossNoise, prec1 = self._forward(noiseInput, noiseLab, epoch, self.modelB_ema)
                #lossNoise_ema, prec1_ema = self._forward(noiseInput, noiseLab, epoch, self.modelB_ema)
                # sample mining
                lossIdx = np.argsort(lossNoise.data.cpu()).cuda()
                noiseInput, noiseLab = noiseInput[0][lossIdx], noiseLab[lossIdx]
                noiseInput, noiseLab = noiseInput[:int(remRate * lossNoise.shape[0]), ...], noiseLab[:int(
                    remRate * lossNoise.shape[0])]
                # save small samples
                # noiseImg = np.asarray(noiseNames)[lossIdx][:int(remRate*lossNoise.shape[0])]
                # self.saveImg(noiseImg, noiseLab) # save image
                # mix update, part assigned and part unassigned
                mixInput, mixLab = [torch.cat([inputs[0], noiseInput])], torch.cat([targets.long(), noiseLab])
                lossMix, precCNNA = self._forward(mixInput, mixLab, epoch, self.modelA)
                lossMix_ema, precCNNA_ema = self._forward(mixInput, mixLab, epoch, self.modelA_ema)
                lossCNNA = lossMix.mean()
                lossCNNA_ema = lossMix_ema.mean()
                lossCNNA_mean = (lossCNNA + lossCNNA_ema)*0.5
                # update
                losses.update(lossCNNA_mean.item(), mixLab.size(0))
                precisions.update(precCNNA, mixLab.size(0))
                # update CNNA
                self.optimizers[0].zero_grad()
                lossCNNA_mean.backward()
                for param in self.modelA.parameters():
                    try:
                        param.grad.data.clamp(-1., 1.)
                    except:
                        continue
                # update modelA
                self.optimizers[0].step()
                
                self._update_ema_variables(self.modelA, self.modelA_ema, self.alpha, epoch*len(self.newDataSet)+i)
                for param in self.modelA_ema.parameters():
                    try:
                        param.grad.data.clamp(-1., 1.)
                    except:
                        continue

            batch_time.update(time.time() - end)
            end = time.time()
            if (i + 1) % self.print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(self.newDataSet),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              precisions.val, precisions.avg))
    
    def _update_ema_variables(self, model, ema_model, alpha, global_step):
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
        
    def _parse_data(self, inputs):
        imgs, fname, pids, _ = inputs
        inputs = [Variable(imgs).cuda()]
        targets = Variable(pids.cuda())
        return inputs, targets, fname  # image and pid

    def _forward(self, inputs, targets, epoch, model):
        outputs = model(*inputs)  # outputs=[x1,x2,x3]
        # x1 triplet loss
        loss_tri, _ = self.criterions[0](outputs[0], targets, epoch)  # feature
        # x2 triplet loss
        loss_global, prec_global = self.criterions[1](outputs[1], targets, epoch)
        return loss_tri + loss_global, prec_global


#CoTrainerAsyMean_3model
# trainer = CoTrainerAsyMean_3model(
            # model, coModel, coModel_outliers, model_ema, coModel_ema, coModel_outliers_ema, train_inliers_loader, train_outliers_loader, unLoader, criterion, optims
        # )
class CoTrainerAsyMean_3model(object):
    def saveImg(self, imgList, gtList):
        import shutil
        import os
        rootDir = self.noiseData.dataset.root
        if os.path.exists('smallLoss'):
            shutil.rmtree('smallLoss')
        os.makedirs('smallLoss')
        for name, pid in zip(imgList, gtList):
            curPath = os.path.join(rootDir, name)
            nameList = name.split('_')
            nameList[0] = str(pid)
            tarPath = os.path.join('smallLoss', '_'.join(nameList))
            shutil.copyfile(curPath, tarPath)

    def __init__(self, model, coModel,coModel_outliers, model_ema, coModel_ema, coModel_outliers_ema, newDataSet, inNoiseDataSet, outNoiseDataSet, criterions, optimizers, alpha=0.999, print_freq=1):
        self.modelA = model
        self.modelB = coModel  # the co-teacher
        self.modelC = coModel_outliers
        self.modelA_ema = model_ema
        self.modelB_ema = coModel_ema
        self.modelC_ema = coModel_outliers_ema
        self.newDataSet = newDataSet
        self.inNoiseData = inNoiseDataSet
        self.outNoiseData = outNoiseDataSet

        self.criterions = criterions
        self.optimizers = optimizers
        self.alpha = alpha
        self.print_freq = print_freq

    def train(self, epoch, remRate=0.2):
        self.modelA.train()
        self.modelB.train()
        self.modelC.train()
        self.modelA_ema.train()
        self.modelB_ema.train()
        self.modelC_ema.train()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()
        #precisions = [AverageMeter(),AverageMeter()]
        end = time.time()

        for i, inputs in enumerate(self.newDataSet):
            data_time.update(time.time() - end)
            inputs, targets, names = self._parse_data(inputs)  # image and pid
            if i % 2 == 0:
                # update CNNC
                lossPure, prec1 = self._forward(inputs, targets, epoch, self.modelA)  # assigned samples
                #lossPure_ema, prec1_ema = self._forward(inputs, targets, epoch, self.modelA_ema)
                # # assigned's easy samples
                lossIdx = np.argsort(lossPure.data.cpu()).cuda()
                #lossIdx_ema = np.argsort(lossPure_ema.data.cpu()).cuda()
                pureInput = [inputs[0][lossIdx[:int(remRate * lossPure.shape[0])], ...]]
                #pureInput_ema = [inputs[0][lossIdx_ema[:int(remRate * lossPure_ema.shape[0])], ...]]
                pureLab = targets[lossIdx[:int(remRate * lossPure.shape[0])]].long()
                #pureLab_ema = targets[lossIdx_ema[:int(remRate * lossPure_ema.shape[0])]].long()
                # loss for cnn C
                lossCNNC, precCNNC = self._forward(pureInput, pureLab, epoch, self.modelC)
                lossCNNC_ema, precCNNC_ema = self._forward(pureInput, pureLab, epoch, self.modelC_ema)
                lossCNNC = lossCNNC.mean()
                lossCNNC_ema = lossCNNC_ema.mean()
                lossCNNC_mean = (lossCNNC + lossCNNC_ema)*0.5
                losses.update(lossCNNC_mean.item(), pureLab.size(0))
                precisions.update(precCNNC, pureLab.size(0))
                self.optimizers[2].zero_grad()
                lossCNNC_mean.backward()
                for param in self.modelC.parameters():#梯度截断
                    try:
                        param.grad.data.clamp(-1., 1.)
                    except:
                        continue
                self.optimizers[2].step()
                
                self._update_ema_variables(self.modelC, self.modelC_ema, self.alpha, epoch*len(self.newDataSet)+i)
                for param in self.modelC_ema.parameters():#梯度截断
                    try:
                        param.grad.data.clamp(-1., 1.)
                    except:
                        continue
                
            else:
                # update CNNA
                try:
                    noiseInput = next(self.outNoiseData)
                except:
                    noiseLoader = iter(self.outNoiseData)
                    noiseInput = next(noiseLoader)
                noiseInput, noiseLab, noiseNames = self._parse_data(noiseInput)
                lossNoise, prec1 = self._forward(noiseInput, noiseLab, epoch, self.modelC)
                #lossNoise_ema, prec1_ema = self._forward(noiseInput, noiseLab, epoch, self.modelB_ema)
                # sample mining
                lossIdx = np.argsort(lossNoise.data.cpu()).cuda()
                noiseInput, noiseLab = noiseInput[0][lossIdx], noiseLab[lossIdx]
                noiseInput, noiseLab = noiseInput[:int(remRate * lossNoise.shape[0]), ...], noiseLab[:int(
                    remRate * lossNoise.shape[0])]
                # save small samples
                # noiseImg = np.asarray(noiseNames)[lossIdx][:int(remRate*lossNoise.shape[0])]
                # self.saveImg(noiseImg, noiseLab) # save image
                # mix update, part assigned and part unassigned
                mixInput, mixLab = [torch.cat([inputs[0], noiseInput])], torch.cat([targets.long(), noiseLab])
                lossMix, precCNNA = self._forward(mixInput, mixLab, epoch, self.modelA)
                lossMix_ema, precCNNA_ema = self._forward(mixInput, mixLab, epoch, self.modelA_ema)
                lossCNNA = lossMix.mean()
                lossCNNA_ema = lossMix_ema.mean()
                lossCNNA_mean = (lossCNNA + lossCNNA_ema)*0.5
                # update
                losses.update(lossCNNA_mean.item(), mixLab.size(0))
                precisions.update(precCNNA, mixLab.size(0))
                # update CNNA
                self.optimizers[0].zero_grad()
                lossCNNA_mean.backward()
                for param in self.modelA.parameters():
                    try:
                        param.grad.data.clamp(-1., 1.)
                    except:
                        continue
                # update modelA
                self.optimizers[0].step()
                
                self._update_ema_variables(self.modelA, self.modelA_ema, self.alpha, epoch*len(self.newDataSet)+i)
                for param in self.modelA_ema.parameters():
                    try:
                        param.grad.data.clamp(-1., 1.)
                    except:
                        continue

            batch_time.update(time.time() - end)
            end = time.time()
            if (i + 1) % self.print_freq == 0:
                print('[AC]Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(self.newDataSet),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              precisions.val, precisions.avg))
                              
        for i, inputs in enumerate(self.newDataSet):
            data_time.update(time.time() - end)
            inputs, targets, names = self._parse_data(inputs)  # image and pid
            if i % 2 == 0:
                # update CNNB
                lossPure, prec1 = self._forward(inputs, targets, epoch, self.modelA)  # assigned samples
                #lossPure_ema, prec1_ema = self._forward(inputs, targets, epoch, self.modelA_ema)
                # # assigned's easy samples
                lossIdx = np.argsort(lossPure.data.cpu()).cuda()
                #lossIdx_ema = np.argsort(lossPure_ema.data.cpu()).cuda()
                pureInput = [inputs[0][lossIdx[:int(remRate * lossPure.shape[0])], ...]]
                #pureInput_ema = [inputs[0][lossIdx_ema[:int(remRate * lossPure_ema.shape[0])], ...]]
                pureLab = targets[lossIdx[:int(remRate * lossPure.shape[0])]].long()
                #pureLab_ema = targets[lossIdx_ema[:int(remRate * lossPure_ema.shape[0])]].long()
                # loss for cnn B
                lossCNNB, precCNNB = self._forward(pureInput, pureLab, epoch, self.modelB)
                lossCNNB_ema, precCNNB_ema = self._forward(pureInput, pureLab, epoch, self.modelB_ema)
                lossCNNB = lossCNNB.mean()
                lossCNNB_ema = lossCNNB_ema.mean()
                lossCNNB_mean = (lossCNNB + lossCNNB_ema)*0.5
                losses.update(lossCNNB_mean.item(), pureLab.size(0))
                precisions.update(precCNNB, pureLab.size(0))
                self.optimizers[1].zero_grad()
                lossCNNB_mean.backward()
                for param in self.modelB.parameters():#梯度截断
                    try:
                        param.grad.data.clamp(-1., 1.)
                    except:
                        continue
                self.optimizers[1].step()
                
                self._update_ema_variables(self.modelB, self.modelB_ema, self.alpha, epoch*len(self.newDataSet)+i)
                for param in self.modelB_ema.parameters():#梯度截断
                    try:
                        param.grad.data.clamp(-1., 1.)
                    except:
                        continue
                
            else:
                # update CNNA
                try:
                    noiseInput = next(self.inNoiseData)
                except:
                    noiseLoader = iter(self.inNoiseData)
                    noiseInput = next(noiseLoader)
                noiseInput, noiseLab, noiseNames = self._parse_data(noiseInput)
                lossNoise, prec1 = self._forward(noiseInput, noiseLab, epoch, self.modelB)
                #lossNoise_ema, prec1_ema = self._forward(noiseInput, noiseLab, epoch, self.modelB_ema)
                # sample mining
                lossIdx = np.argsort(lossNoise.data.cpu()).cuda()
                noiseInput, noiseLab = noiseInput[0][lossIdx], noiseLab[lossIdx]
                noiseInput, noiseLab = noiseInput[:int(remRate * lossNoise.shape[0]), ...], noiseLab[:int(
                    remRate * lossNoise.shape[0])]
                # save small samples
                # noiseImg = np.asarray(noiseNames)[lossIdx][:int(remRate*lossNoise.shape[0])]
                # self.saveImg(noiseImg, noiseLab) # save image
                # mix update, part assigned and part unassigned
                mixInput, mixLab = [torch.cat([inputs[0], noiseInput])], torch.cat([targets.long(), noiseLab])
                lossMix, precCNNA = self._forward(mixInput, mixLab, epoch, self.modelA)
                lossMix_ema, precCNNA_ema = self._forward(mixInput, mixLab, epoch, self.modelA_ema)
                lossCNNA = lossMix.mean()
                lossCNNA_ema = lossMix_ema.mean()
                lossCNNA_mean = (lossCNNA + lossCNNA_ema)*0.5
                # update
                losses.update(lossCNNA_mean.item(), mixLab.size(0))
                precisions.update(precCNNA, mixLab.size(0))
                # update CNNA
                self.optimizers[0].zero_grad()
                lossCNNA_mean.backward()
                for param in self.modelA.parameters():
                    try:
                        param.grad.data.clamp(-1., 1.)
                    except:
                        continue
                # update modelA
                self.optimizers[0].step()
                
                self._update_ema_variables(self.modelA, self.modelA_ema, self.alpha, epoch*len(self.newDataSet)+i)
                for param in self.modelA_ema.parameters():
                    try:
                        param.grad.data.clamp(-1., 1.)
                    except:
                        continue

            batch_time.update(time.time() - end)
            end = time.time()
            if (i + 1) % self.print_freq == 0:
                print('[AB]Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(self.newDataSet),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              precisions.val, precisions.avg))
    
    def _update_ema_variables(self, model, ema_model, alpha, global_step):
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
        
    def _parse_data(self, inputs):
        imgs, fname, pids, _ = inputs
        inputs = [Variable(imgs).cuda()]
        targets = Variable(pids.cuda())
        return inputs, targets, fname  # image and pid

    def _forward(self, inputs, targets, epoch, model):
        outputs = model(*inputs)  # outputs=[x1,x2,x3]
        # x1 triplet loss
        loss_tri, _ = self.criterions[0](outputs[0], targets, epoch)  # feature
        # x2 triplet loss
        loss_global, prec_global = self.criterions[1](outputs[1], targets, epoch)
        return loss_tri + loss_global, prec_global


# trainer = CoTrainerAsyMean_4model(
            # model, coModel, co2Model, co3Model, model_ema, coModel_ema, co2Model_ema, co3Model_ema, train_in3_loader, train_out3_loader, train_out2_loader, unLoader, criterion, optims
        # )
class CoTrainerAsyMean_4model(object):
    def saveImg(self, imgList, gtList):
        import shutil
        import os
        rootDir = self.noiseData.dataset.root
        if os.path.exists('smallLoss'):
            shutil.rmtree('smallLoss')
        os.makedirs('smallLoss')
        for name, pid in zip(imgList, gtList):
            curPath = os.path.join(rootDir, name)
            nameList = name.split('_')
            nameList[0] = str(pid)
            tarPath = os.path.join('smallLoss', '_'.join(nameList))
            shutil.copyfile(curPath, tarPath)

    def __init__(self, model, coModel, co2Model, co3Model, model_ema, coModel_ema, co2Model_ema, co3Model_ema, train_in3_loader, train_out3_loader, train_out2_loader, unLoader, criterions, optimizers, alpha=0.999, print_freq=1):
        self.modelA = model
        self.modelB = coModel  # the co-teacher
        self.modelC = co2Model
        self.modelD = co3Model
        self.modelA_ema = model_ema
        self.modelB_ema = coModel_ema  
        self.modelC_ema = co2Model_ema
        self.modelD_ema = co3Model_ema
        #self.noiseData = noiseDataSet
        self.train_in3_data = train_in3_loader
        self.train_out3_data = train_out3_loader
        self.train_out2_data = train_out2_loader
        self.train_out_data = unLoader
        self.criterions = criterions
        self.optimizers = optimizers
        self.alpha = alpha
        self.print_freq = print_freq

    def train(self, epoch, remRate=0.2):
        self.modelA.train()
        self.modelB.train()
        self.modelC.train()
        self.modelD.train()
        self.modelA_ema.train()
        self.modelB_ema.train()
        self.modelC_ema.train()
        self.modelD_ema.train()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()
        end = time.time()
        
        
        for i, inputs in enumerate(self.train_in3_data):
            data_time.update(time.time() - end)
            inputs, targets, names = self._parse_data(inputs)  # image and pid
            #print('inputs.type: {} ,inputs.size: {} '.format(type(inputs), len(inputs)))
            #print('inputs.type: {} ,inputs[0].size: {} '.format(type(inputs), len(inputs[0])))
            #print('inputs.type: {} ,inputs[0][0].size: {} '.format(type(inputs), len(inputs[0][0])))
            #print('inputs.type: {} ,inputs[1].size: {} '.format(type(inputs), len(inputs[1])))
            # print(inputs[0])
            #print(np.array(inputs).shape)
            if i % 2 == 0:
                # update CNND
                lossPure, prec1 = self._forward(inputs, targets, epoch, self.modelA)  # assigned samples
                # # assigned's easy samples
                lossIdx = np.argsort(lossPure.data.cpu()).cuda()
                pureInput = [inputs[0][lossIdx[:int(remRate * lossPure.shape[0])], ...]]
                pureLab = targets[lossIdx[:int(remRate * lossPure.shape[0])]].long()
                # loss for cnn D
                lossCNND, precCNND = self._forward(pureInput, pureLab, epoch, self.modelD)
                lossCNND_ema, precCNND_ema = self._forward(pureInput, pureLab, epoch, self.modelD_ema)
                lossCNND = lossCNND.mean()
                lossCNND_ema = lossCNND_ema.mean()
                lossCNND_mean = (lossCNND + lossCNND_ema)*0.5
                losses.update(lossCNND_mean.item(), pureLab.size(0))
                precisions.update(precCNND, pureLab.size(0))
                self.optimizers[3].zero_grad()
                lossCNND_mean.backward()
                for param in self.modelD.parameters():
                    try:
                        param.grad.data.clamp(-1., 1.)
                    except:
                        continue
                self.optimizers[3].step()
                
                self._update_ema_variables(self.modelD, self.modelD_ema, self.alpha, epoch*len(self.train_in3_data)+i)
                for param in self.modelD_ema.parameters():#梯度截断
                    try:
                        param.grad.data.clamp(-1., 1.)
                    except:
                        continue
            else:
                # update CNNA
                try:
                    noiseInput = next(self.train_out_data)
                except:
                    noiseLoader = iter(self.train_out_data)
                    noiseInput = next(noiseLoader)
                noiseInput, noiseLab, noiseNames = self._parse_data(noiseInput)
                lossNoise, prec1 = self._forward(noiseInput, noiseLab, epoch, self.modelD)
                # sample mining
                lossIdx = np.argsort(lossNoise.data.cpu()).cuda()
                noiseInput, noiseLab = noiseInput[0][lossIdx], noiseLab[lossIdx]
                noiseInput, noiseLab = noiseInput[:int(remRate * lossNoise.shape[0]), ...], noiseLab[:int(
                    remRate * lossNoise.shape[0])]
                # save small samples
                # noiseImg = np.asarray(noiseNames)[lossIdx][:int(remRate*lossNoise.shape[0])]
                # self.saveImg(noiseImg, noiseLab) # save image
                # mix update, part assigned and part unassigned
                mixInput, mixLab = [torch.cat([inputs[0], noiseInput])], torch.cat([targets.long(), noiseLab])
                lossMix, precCNNA = self._forward(mixInput, mixLab, epoch, self.modelA)
                lossMix_ema, precCNNA_ema = self._forward(mixInput, mixLab, epoch, self.modelA_ema)
                lossCNNA = lossMix.mean()
                lossCNNA_ema = lossMix_ema.mean()
                lossCNNA_mean = (lossCNNA + lossCNNA_ema)*0.5
                # update
                losses.update(lossCNNA_mean.item(), mixLab.size(0))
                precisions.update(precCNNA, mixLab.size(0))
                # update CNNA
                self.optimizers[0].zero_grad()
                lossCNNA_mean.backward()
                for param in self.modelA.parameters():
                    try:
                        param.grad.data.clamp(-1., 1.)
                    except:
                        continue
                # update modelA
                self.optimizers[0].step()
                
                self._update_ema_variables(self.modelA, self.modelA_ema, self.alpha, epoch*len(self.train_in3_data)+i)
                for param in self.modelA_ema.parameters():
                    try:
                        param.grad.data.clamp(-1., 1.)
                    except:
                        continue

            batch_time.update(time.time() - end)
            end = time.time()
            if (i + 1) % self.print_freq == 0:
                print('[AD]Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(self.train_in3_data),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              precisions.val, precisions.avg))
        
        
        for i, inputs in enumerate(self.train_in3_data):
            data_time.update(time.time() - end)
            inputs, targets, names = self._parse_data(inputs)  # image and pid
            #print('inputs.type: {} ,inputs.size: {} '.format(type(inputs), len(inputs)))
            #print('inputs.type: {} ,inputs[0].size: {} '.format(type(inputs), len(inputs[0])))
            #print('inputs.type: {} ,inputs[0][0].size: {} '.format(type(inputs), len(inputs[0][0])))
            #print('inputs.type: {} ,inputs[1].size: {} '.format(type(inputs), len(inputs[1])))
            # print(inputs[0])
            #print(np.array(inputs).shape)
            if i % 2 == 0:
                # update CNNC
                lossPure, prec1 = self._forward(inputs, targets, epoch, self.modelA)  # assigned samples
                # # assigned's easy samples
                lossIdx = np.argsort(lossPure.data.cpu()).cuda()
                pureInput = [inputs[0][lossIdx[:int(remRate * lossPure.shape[0])], ...]]
                pureLab = targets[lossIdx[:int(remRate * lossPure.shape[0])]].long()
                # loss for cnn C
                lossCNNC, precCNNC = self._forward(pureInput, pureLab, epoch, self.modelC)
                lossCNNC_ema, precCNNC_ema = self._forward(pureInput, pureLab, epoch, self.modelC_ema)
                lossCNNC = lossCNNC.mean()
                lossCNNC_ema = lossCNNC_ema.mean()
                lossCNNC_mean = (lossCNNC + lossCNNC_ema)*0.5
                losses.update(lossCNNC_mean.item(), pureLab.size(0))
                precisions.update(precCNNC, pureLab.size(0))
                self.optimizers[2].zero_grad()
                lossCNNC_mean.backward()
                for param in self.modelC.parameters():
                    try:
                        param.grad.data.clamp(-1., 1.)
                    except:
                        continue
                self.optimizers[2].step()
                
                self._update_ema_variables(self.modelC, self.modelC_ema, self.alpha, epoch*len(self.train_in3_data)+i)
                for param in self.modelC_ema.parameters():#梯度截断
                    try:
                        param.grad.data.clamp(-1., 1.)
                    except:
                        continue
                
            else:
                # update CNNA
                try:
                    noiseInput = next(self.train_out2_data)
                except:
                    noiseLoader = iter(self.train_out2_data)
                    noiseInput = next(noiseLoader)
                noiseInput, noiseLab, noiseNames = self._parse_data(noiseInput)
                lossNoise, prec1 = self._forward(noiseInput, noiseLab, epoch, self.modelC)
                # sample mining
                lossIdx = np.argsort(lossNoise.data.cpu()).cuda()
                noiseInput, noiseLab = noiseInput[0][lossIdx], noiseLab[lossIdx]
                noiseInput, noiseLab = noiseInput[:int(remRate * lossNoise.shape[0]), ...], noiseLab[:int(
                    remRate * lossNoise.shape[0])]
                # save small samples
                # noiseImg = np.asarray(noiseNames)[lossIdx][:int(remRate*lossNoise.shape[0])]
                # self.saveImg(noiseImg, noiseLab) # save image
                # mix update, part assigned and part unassigned
                mixInput, mixLab = [torch.cat([inputs[0], noiseInput])], torch.cat([targets.long(), noiseLab])
                lossMix, precCNNA = self._forward(mixInput, mixLab, epoch, self.modelA)
                lossMix_ema, precCNNA_ema = self._forward(mixInput, mixLab, epoch, self.modelA_ema)
                lossCNNA = lossMix.mean()
                lossCNNA_ema = lossMix_ema.mean()
                lossCNNA_mean = (lossCNNA + lossCNNA_ema)*0.5
                # update
                losses.update(lossCNNA_mean.item(), mixLab.size(0))
                precisions.update(precCNNA, mixLab.size(0))
                # update CNNA
                self.optimizers[0].zero_grad()
                lossCNNA_mean.backward()
                for param in self.modelA.parameters():
                    try:
                        param.grad.data.clamp(-1., 1.)
                    except:
                        continue
                # update modelA
                self.optimizers[0].step()
                
                self._update_ema_variables(self.modelA, self.modelA_ema, self.alpha, epoch*len(self.train_in3_data)+i)
                for param in self.modelA_ema.parameters():
                    try:
                        param.grad.data.clamp(-1., 1.)
                    except:
                        continue

            batch_time.update(time.time() - end)
            end = time.time()
            if (i + 1) % self.print_freq == 0:
                print('[AC]Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(self.train_in3_data),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              precisions.val, precisions.avg))
        
        for i, inputs in enumerate(self.train_in3_data):
            data_time.update(time.time() - end)
            inputs, targets, names = self._parse_data(inputs)  # image and pid
            #print('inputs.type: {} ,inputs.size: {} '.format(type(inputs), len(inputs)))
            #print('inputs.type: {} ,inputs[0].size: {} '.format(type(inputs), len(inputs[0])))
            #print('inputs.type: {} ,inputs[0][0].size: {} '.format(type(inputs), len(inputs[0][0])))
            #print('inputs.type: {} ,inputs[1].size: {} '.format(type(inputs), len(inputs[1])))
            # print(inputs[0])
            #print(np.array(inputs).shape)
            if i % 2 == 0:
                # update CNNB
                lossPure, prec1 = self._forward(inputs, targets, epoch, self.modelA)  # assigned samples
                # # assigned's easy samples
                lossIdx = np.argsort(lossPure.data.cpu()).cuda()
                pureInput = [inputs[0][lossIdx[:int(remRate * lossPure.shape[0])], ...]]
                pureLab = targets[lossIdx[:int(remRate * lossPure.shape[0])]].long()
                # loss for cnn B
                lossCNNB, precCNNB = self._forward(pureInput, pureLab, epoch, self.modelB)
                lossCNNB_ema, precCNNB_ema = self._forward(pureInput, pureLab, epoch, self.modelB_ema)
                lossCNNB = lossCNNB.mean()
                lossCNNB_ema = lossCNNB_ema.mean()
                lossCNNB_mean = (lossCNNB + lossCNNB_ema)*0.5
                losses.update(lossCNNB_mean.item(), pureLab.size(0))
                precisions.update(precCNNB, pureLab.size(0))
                self.optimizers[1].zero_grad()
                lossCNNB_mean.backward()
                for param in self.modelB.parameters():
                    try:
                        param.grad.data.clamp(-1., 1.)
                    except:
                        continue
                self.optimizers[1].step()
                
                self._update_ema_variables(self.modelB, self.modelB_ema, self.alpha, epoch*len(self.train_in3_data)+i)
                for param in self.modelB_ema.parameters():#梯度截断
                    try:
                        param.grad.data.clamp(-1., 1.)
                    except:
                        continue
            else:
                # update CNNA
                try:
                    noiseInput = next(self.train_out3_data)
                except:
                    noiseLoader = iter(self.train_out3_data)
                    noiseInput = next(noiseLoader)
                noiseInput, noiseLab, noiseNames = self._parse_data(noiseInput)
                lossNoise, prec1 = self._forward(noiseInput, noiseLab, epoch, self.modelB)
                # sample mining
                lossIdx = np.argsort(lossNoise.data.cpu()).cuda()
                noiseInput, noiseLab = noiseInput[0][lossIdx], noiseLab[lossIdx]
                noiseInput, noiseLab = noiseInput[:int(remRate * lossNoise.shape[0]), ...], noiseLab[:int(
                    remRate * lossNoise.shape[0])]
                # save small samples
                # noiseImg = np.asarray(noiseNames)[lossIdx][:int(remRate*lossNoise.shape[0])]
                # self.saveImg(noiseImg, noiseLab) # save image
                # mix update, part assigned and part unassigned
                mixInput, mixLab = [torch.cat([inputs[0], noiseInput])], torch.cat([targets.long(), noiseLab])
                lossMix, precCNNA = self._forward(mixInput, mixLab, epoch, self.modelA)
                lossMix_ema, precCNNA_ema = self._forward(mixInput, mixLab, epoch, self.modelA_ema)
                lossCNNA = lossMix.mean()
                lossCNNA_ema = lossMix_ema.mean()
                lossCNNA_mean = (lossCNNA + lossCNNA_ema)*0.5
                # update
                losses.update(lossCNNA_mean.item(), mixLab.size(0))
                precisions.update(precCNNA, mixLab.size(0))
                # update CNNA
                self.optimizers[0].zero_grad()
                lossCNNA_mean.backward()
                for param in self.modelA.parameters():
                    try:
                        param.grad.data.clamp(-1., 1.)
                    except:
                        continue
                # update modelA
                self.optimizers[0].step()
                
                self._update_ema_variables(self.modelA, self.modelA_ema, self.alpha, epoch*len(self.train_in3_data)+i)
                for param in self.modelA_ema.parameters():
                    try:
                        param.grad.data.clamp(-1., 1.)
                    except:
                        continue


            batch_time.update(time.time() - end)
            end = time.time()
            if (i + 1) % self.print_freq == 0:
                print('[AB]Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(self.train_in3_data),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              precisions.val, precisions.avg))
        
    def _update_ema_variables(self, model, ema_model, alpha, global_step):
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    def _parse_data(self, inputs):
        imgs, fname, pids, _ = inputs
        inputs = [Variable(imgs).cuda()]
        targets = Variable(pids.cuda())
        return inputs, targets, fname  # image and pid

    def _forward(self, inputs, targets, epoch, model):
        outputs = model(*inputs)  # outputs=[x1,x2,x3]
        # x1 triplet loss
        loss_tri, _ = self.criterions[0](outputs[0], targets, epoch)  # feature
        # x2 triplet loss
        loss_global, prec_global = self.criterions[1](outputs[1], targets, epoch)
        return loss_tri + loss_global, prec_global

# trainer = CoTrainerAsyMean_5model(
    # model, coModel, co2Model, co3Model, co4Model, model_ema, coModel_ema, co2Model_ema, co3Model_ema, co4Model_ema, train_in4_loader, train_out4_loader, train_out3_loader, train_out2_loader, unLoader, criterion, optims
# )
class CoTrainerAsyMean_5model(object):
    def saveImg(self, imgList, gtList):
        import shutil
        import os
        rootDir = self.noiseData.dataset.root
        if os.path.exists('smallLoss'):
            shutil.rmtree('smallLoss')
        os.makedirs('smallLoss')
        for name, pid in zip(imgList, gtList):
            curPath = os.path.join(rootDir, name)
            nameList = name.split('_')
            nameList[0] = str(pid)
            tarPath = os.path.join('smallLoss', '_'.join(nameList))
            shutil.copyfile(curPath, tarPath)

    def __init__(self, model, coModel, co2Model, co3Model, co4Model, model_ema, coModel_ema, co2Model_ema, co3Model_ema, co4Model_ema, train_in4_loader, train_out4_loader, train_out3_loader, train_out2_loader, unLoader, criterions, optimizers, alpha=0.999, print_freq=1):
        self.modelA = model
        self.modelB = coModel  # the co-teacher
        self.modelC = co2Model
        self.modelD = co3Model
        self.modelE = co4Model
        self.modelA_ema = model_ema
        self.modelB_ema = coModel_ema  
        self.modelC_ema = co2Model_ema
        self.modelD_ema = co3Model_ema
        self.modelE_ema = co4Model_ema
        #self.noiseData = noiseDataSet
        self.train_in4_data = train_in4_loader
        self.train_out4_data = train_out4_loader
        self.train_out3_data = train_out3_loader
        self.train_out2_data = train_out2_loader
        self.train_out_data = unLoader
        self.criterions = criterions
        self.optimizers = optimizers
        self.alpha = alpha
        self.print_freq = print_freq

    def train(self, epoch, remRate=0.2):
        self.modelA.train()
        self.modelB.train()
        self.modelC.train()
        self.modelD.train()
        self.modelE.train()
        self.modelA_ema.train()
        self.modelB_ema.train()
        self.modelC_ema.train()
        self.modelD_ema.train()
        self.modelE_ema.train()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()
        end = time.time()
        
        for i, inputs in enumerate(self.train_in4_data):
            data_time.update(time.time() - end)
            inputs, targets, names = self._parse_data(inputs)  # image and pid
            #print('inputs.type: {} ,inputs.size: {} '.format(type(inputs), len(inputs)))
            #print('inputs.type: {} ,inputs[0].size: {} '.format(type(inputs), len(inputs[0])))
            #print('inputs.type: {} ,inputs[0][0].size: {} '.format(type(inputs), len(inputs[0][0])))
            #print('inputs.type: {} ,inputs[1].size: {} '.format(type(inputs), len(inputs[1])))
            # print(inputs[0])
            #print(np.array(inputs).shape)
            if i % 2 == 0:
                # update CNNE
                lossPure, prec1 = self._forward(inputs, targets, epoch, self.modelA)  # assigned samples
                # # assigned's easy samples
                lossIdx = np.argsort(lossPure.data.cpu()).cuda()
                pureInput = [inputs[0][lossIdx[:int(remRate * lossPure.shape[0])], ...]]
                pureLab = targets[lossIdx[:int(remRate * lossPure.shape[0])]].long()
                # loss for cnn D
                lossCNNE, precCNNE = self._forward(pureInput, pureLab, epoch, self.modelE)
                lossCNNE_ema, precCNNE_ema = self._forward(pureInput, pureLab, epoch, self.modelE_ema)
                lossCNNE = lossCNNE.mean()
                lossCNNE_ema = lossCNNE_ema.mean()
                lossCNNE_mean = (lossCNNE + lossCNNE_ema)*0.5
                losses.update(lossCNNE_mean.item(), pureLab.size(0))
                precisions.update(precCNNE, pureLab.size(0))
                self.optimizers[4].zero_grad()
                lossCNNE_mean.backward()
                for param in self.modelE.parameters():
                    try:
                        param.grad.data.clamp(-1., 1.)
                    except:
                        continue
                self.optimizers[4].step()
                
                self._update_ema_variables(self.modelE, self.modelE_ema, self.alpha, epoch*len(self.train_in4_data)+i)
                for param in self.modelE_ema.parameters():#梯度截断
                    try:
                        param.grad.data.clamp(-1., 1.)
                    except:
                        continue
            else:
                # update CNNA
                try:
                    noiseInput = next(self.train_out_data)
                except:
                    noiseLoader = iter(self.train_out_data)
                    noiseInput = next(noiseLoader)
                noiseInput, noiseLab, noiseNames = self._parse_data(noiseInput)
                lossNoise, prec1 = self._forward(noiseInput, noiseLab, epoch, self.modelE)
                # sample mining
                lossIdx = np.argsort(lossNoise.data.cpu()).cuda()
                noiseInput, noiseLab = noiseInput[0][lossIdx], noiseLab[lossIdx]
                noiseInput, noiseLab = noiseInput[:int(remRate * lossNoise.shape[0]), ...], noiseLab[:int(
                    remRate * lossNoise.shape[0])]
                # save small samples
                # noiseImg = np.asarray(noiseNames)[lossIdx][:int(remRate*lossNoise.shape[0])]
                # self.saveImg(noiseImg, noiseLab) # save image
                # mix update, part assigned and part unassigned
                mixInput, mixLab = [torch.cat([inputs[0], noiseInput])], torch.cat([targets.long(), noiseLab])
                lossMix, precCNNA = self._forward(mixInput, mixLab, epoch, self.modelA)
                lossMix_ema, precCNNA_ema = self._forward(mixInput, mixLab, epoch, self.modelA_ema)
                lossCNNA = lossMix.mean()
                lossCNNA_ema = lossMix_ema.mean()
                lossCNNA_mean = (lossCNNA + lossCNNA_ema)*0.5
                # update
                losses.update(lossCNNA_mean.item(), mixLab.size(0))
                precisions.update(precCNNA, mixLab.size(0))
                # update CNNA
                self.optimizers[0].zero_grad()
                lossCNNA_mean.backward()
                for param in self.modelA.parameters():
                    try:
                        param.grad.data.clamp(-1., 1.)
                    except:
                        continue
                # update modelA
                self.optimizers[0].step()
                
                self._update_ema_variables(self.modelA, self.modelA_ema, self.alpha, epoch*len(self.train_in4_data)+i)
                for param in self.modelA_ema.parameters():
                    try:
                        param.grad.data.clamp(-1., 1.)
                    except:
                        continue

            batch_time.update(time.time() - end)
            end = time.time()
            if (i + 1) % self.print_freq == 0:
                print('[AE]Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(self.train_in4_data),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              precisions.val, precisions.avg))
        
        
        for i, inputs in enumerate(self.train_in4_data):
            data_time.update(time.time() - end)
            inputs, targets, names = self._parse_data(inputs)  # image and pid
            #print('inputs.type: {} ,inputs.size: {} '.format(type(inputs), len(inputs)))
            #print('inputs.type: {} ,inputs[0].size: {} '.format(type(inputs), len(inputs[0])))
            #print('inputs.type: {} ,inputs[0][0].size: {} '.format(type(inputs), len(inputs[0][0])))
            #print('inputs.type: {} ,inputs[1].size: {} '.format(type(inputs), len(inputs[1])))
            # print(inputs[0])
            #print(np.array(inputs).shape)
            if i % 2 == 0:
                # update CNND
                lossPure, prec1 = self._forward(inputs, targets, epoch, self.modelA)  # assigned samples
                # # assigned's easy samples
                lossIdx = np.argsort(lossPure.data.cpu()).cuda()
                pureInput = [inputs[0][lossIdx[:int(remRate * lossPure.shape[0])], ...]]
                pureLab = targets[lossIdx[:int(remRate * lossPure.shape[0])]].long()
                # loss for cnn D
                lossCNND, precCNND = self._forward(pureInput, pureLab, epoch, self.modelD)
                lossCNND_ema, precCNND_ema = self._forward(pureInput, pureLab, epoch, self.modelD_ema)
                lossCNND = lossCNND.mean()
                lossCNND_ema = lossCNND_ema.mean()
                lossCNND_mean = (lossCNND + lossCNND_ema)*0.5
                losses.update(lossCNND_mean.item(), pureLab.size(0))
                precisions.update(precCNND, pureLab.size(0))
                self.optimizers[3].zero_grad()
                lossCNND_mean.backward()
                for param in self.modelD.parameters():
                    try:
                        param.grad.data.clamp(-1., 1.)
                    except:
                        continue
                self.optimizers[3].step()
                
                self._update_ema_variables(self.modelD, self.modelD_ema, self.alpha, epoch*len(self.train_in4_data)+i)
                for param in self.modelD_ema.parameters():#梯度截断
                    try:
                        param.grad.data.clamp(-1., 1.)
                    except:
                        continue
            else:
                # update CNNA
                try:
                    noiseInput = next(self.train_out2_data)
                except:
                    noiseLoader = iter(self.train_out2_data)
                    noiseInput = next(noiseLoader)
                noiseInput, noiseLab, noiseNames = self._parse_data(noiseInput)
                lossNoise, prec1 = self._forward(noiseInput, noiseLab, epoch, self.modelD)
                # sample mining
                lossIdx = np.argsort(lossNoise.data.cpu()).cuda()
                noiseInput, noiseLab = noiseInput[0][lossIdx], noiseLab[lossIdx]
                noiseInput, noiseLab = noiseInput[:int(remRate * lossNoise.shape[0]), ...], noiseLab[:int(
                    remRate * lossNoise.shape[0])]
                # save small samples
                # noiseImg = np.asarray(noiseNames)[lossIdx][:int(remRate*lossNoise.shape[0])]
                # self.saveImg(noiseImg, noiseLab) # save image
                # mix update, part assigned and part unassigned
                mixInput, mixLab = [torch.cat([inputs[0], noiseInput])], torch.cat([targets.long(), noiseLab])
                lossMix, precCNNA = self._forward(mixInput, mixLab, epoch, self.modelA)
                lossMix_ema, precCNNA_ema = self._forward(mixInput, mixLab, epoch, self.modelA_ema)
                lossCNNA = lossMix.mean()
                lossCNNA_ema = lossMix_ema.mean()
                lossCNNA_mean = (lossCNNA + lossCNNA_ema)*0.5
                # update
                losses.update(lossCNNA_mean.item(), mixLab.size(0))
                precisions.update(precCNNA, mixLab.size(0))
                # update CNNA
                self.optimizers[0].zero_grad()
                lossCNNA_mean.backward()
                for param in self.modelA.parameters():
                    try:
                        param.grad.data.clamp(-1., 1.)
                    except:
                        continue
                # update modelA
                self.optimizers[0].step()
                
                self._update_ema_variables(self.modelA, self.modelA_ema, self.alpha, epoch*len(self.train_in4_data)+i)
                for param in self.modelA_ema.parameters():
                    try:
                        param.grad.data.clamp(-1., 1.)
                    except:
                        continue

            batch_time.update(time.time() - end)
            end = time.time()
            if (i + 1) % self.print_freq == 0:
                print('[AD]Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(self.train_in4_data),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              precisions.val, precisions.avg))
        
        
        for i, inputs in enumerate(self.train_in4_data):
            data_time.update(time.time() - end)
            inputs, targets, names = self._parse_data(inputs)  # image and pid
            #print('inputs.type: {} ,inputs.size: {} '.format(type(inputs), len(inputs)))
            #print('inputs.type: {} ,inputs[0].size: {} '.format(type(inputs), len(inputs[0])))
            #print('inputs.type: {} ,inputs[0][0].size: {} '.format(type(inputs), len(inputs[0][0])))
            #print('inputs.type: {} ,inputs[1].size: {} '.format(type(inputs), len(inputs[1])))
            # print(inputs[0])
            #print(np.array(inputs).shape)
            if i % 2 == 0:
                # update CNNC
                lossPure, prec1 = self._forward(inputs, targets, epoch, self.modelA)  # assigned samples
                # # assigned's easy samples
                lossIdx = np.argsort(lossPure.data.cpu()).cuda()
                pureInput = [inputs[0][lossIdx[:int(remRate * lossPure.shape[0])], ...]]
                pureLab = targets[lossIdx[:int(remRate * lossPure.shape[0])]].long()
                # loss for cnn C
                lossCNNC, precCNNC = self._forward(pureInput, pureLab, epoch, self.modelC)
                lossCNNC_ema, precCNNC_ema = self._forward(pureInput, pureLab, epoch, self.modelC_ema)
                lossCNNC = lossCNNC.mean()
                lossCNNC_ema = lossCNNC_ema.mean()
                lossCNNC_mean = (lossCNNC + lossCNNC_ema)*0.5
                losses.update(lossCNNC_mean.item(), pureLab.size(0))
                precisions.update(precCNNC, pureLab.size(0))
                self.optimizers[2].zero_grad()
                lossCNNC_mean.backward()
                for param in self.modelC.parameters():
                    try:
                        param.grad.data.clamp(-1., 1.)
                    except:
                        continue
                self.optimizers[2].step()
                
                self._update_ema_variables(self.modelC, self.modelC_ema, self.alpha, epoch*len(self.train_in4_data)+i)
                for param in self.modelC_ema.parameters():#梯度截断
                    try:
                        param.grad.data.clamp(-1., 1.)
                    except:
                        continue
                
            else:
                # update CNNA
                try:
                    noiseInput = next(self.train_out3_data)
                except:
                    noiseLoader = iter(self.train_out3_data)
                    noiseInput = next(noiseLoader)
                noiseInput, noiseLab, noiseNames = self._parse_data(noiseInput)
                lossNoise, prec1 = self._forward(noiseInput, noiseLab, epoch, self.modelC)
                # sample mining
                lossIdx = np.argsort(lossNoise.data.cpu()).cuda()
                noiseInput, noiseLab = noiseInput[0][lossIdx], noiseLab[lossIdx]
                noiseInput, noiseLab = noiseInput[:int(remRate * lossNoise.shape[0]), ...], noiseLab[:int(
                    remRate * lossNoise.shape[0])]
                # save small samples
                # noiseImg = np.asarray(noiseNames)[lossIdx][:int(remRate*lossNoise.shape[0])]
                # self.saveImg(noiseImg, noiseLab) # save image
                # mix update, part assigned and part unassigned
                mixInput, mixLab = [torch.cat([inputs[0], noiseInput])], torch.cat([targets.long(), noiseLab])
                lossMix, precCNNA = self._forward(mixInput, mixLab, epoch, self.modelA)
                lossMix_ema, precCNNA_ema = self._forward(mixInput, mixLab, epoch, self.modelA_ema)
                lossCNNA = lossMix.mean()
                lossCNNA_ema = lossMix_ema.mean()
                lossCNNA_mean = (lossCNNA + lossCNNA_ema)*0.5
                # update
                losses.update(lossCNNA_mean.item(), mixLab.size(0))
                precisions.update(precCNNA, mixLab.size(0))
                # update CNNA
                self.optimizers[0].zero_grad()
                lossCNNA_mean.backward()
                for param in self.modelA.parameters():
                    try:
                        param.grad.data.clamp(-1., 1.)
                    except:
                        continue
                # update modelA
                self.optimizers[0].step()
                
                self._update_ema_variables(self.modelA, self.modelA_ema, self.alpha, epoch*len(self.train_in4_data)+i)
                for param in self.modelA_ema.parameters():
                    try:
                        param.grad.data.clamp(-1., 1.)
                    except:
                        continue

            batch_time.update(time.time() - end)
            end = time.time()
            if (i + 1) % self.print_freq == 0:
                print('[AC]Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(self.train_in4_data),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              precisions.val, precisions.avg))
        
        for i, inputs in enumerate(self.train_in4_data):
            data_time.update(time.time() - end)
            inputs, targets, names = self._parse_data(inputs)  # image and pid
            #print('inputs.type: {} ,inputs.size: {} '.format(type(inputs), len(inputs)))
            #print('inputs.type: {} ,inputs[0].size: {} '.format(type(inputs), len(inputs[0])))
            #print('inputs.type: {} ,inputs[0][0].size: {} '.format(type(inputs), len(inputs[0][0])))
            #print('inputs.type: {} ,inputs[1].size: {} '.format(type(inputs), len(inputs[1])))
            # print(inputs[0])
            #print(np.array(inputs).shape)
            if i % 2 == 0:
                # update CNNB
                lossPure, prec1 = self._forward(inputs, targets, epoch, self.modelA)  # assigned samples
                # # assigned's easy samples
                lossIdx = np.argsort(lossPure.data.cpu()).cuda()
                pureInput = [inputs[0][lossIdx[:int(remRate * lossPure.shape[0])], ...]]
                pureLab = targets[lossIdx[:int(remRate * lossPure.shape[0])]].long()
                # loss for cnn B
                lossCNNB, precCNNB = self._forward(pureInput, pureLab, epoch, self.modelB)
                lossCNNB_ema, precCNNB_ema = self._forward(pureInput, pureLab, epoch, self.modelB_ema)
                lossCNNB = lossCNNB.mean()
                lossCNNB_ema = lossCNNB_ema.mean()
                lossCNNB_mean = (lossCNNB + lossCNNB_ema)*0.5
                losses.update(lossCNNB_mean.item(), pureLab.size(0))
                precisions.update(precCNNB, pureLab.size(0))
                self.optimizers[1].zero_grad()
                lossCNNB_mean.backward()
                for param in self.modelB.parameters():
                    try:
                        param.grad.data.clamp(-1., 1.)
                    except:
                        continue
                self.optimizers[1].step()
                
                self._update_ema_variables(self.modelB, self.modelB_ema, self.alpha, epoch*len(self.train_in4_data)+i)
                for param in self.modelB_ema.parameters():#梯度截断
                    try:
                        param.grad.data.clamp(-1., 1.)
                    except:
                        continue
            else:
                # update CNNA
                try:
                    noiseInput = next(self.train_out4_data)
                except:
                    noiseLoader = iter(self.train_out4_data)
                    noiseInput = next(noiseLoader)
                noiseInput, noiseLab, noiseNames = self._parse_data(noiseInput)
                lossNoise, prec1 = self._forward(noiseInput, noiseLab, epoch, self.modelB)
                # sample mining
                lossIdx = np.argsort(lossNoise.data.cpu()).cuda()
                noiseInput, noiseLab = noiseInput[0][lossIdx], noiseLab[lossIdx]
                noiseInput, noiseLab = noiseInput[:int(remRate * lossNoise.shape[0]), ...], noiseLab[:int(
                    remRate * lossNoise.shape[0])]
                # save small samples
                # noiseImg = np.asarray(noiseNames)[lossIdx][:int(remRate*lossNoise.shape[0])]
                # self.saveImg(noiseImg, noiseLab) # save image
                # mix update, part assigned and part unassigned
                mixInput, mixLab = [torch.cat([inputs[0], noiseInput])], torch.cat([targets.long(), noiseLab])
                lossMix, precCNNA = self._forward(mixInput, mixLab, epoch, self.modelA)
                lossMix_ema, precCNNA_ema = self._forward(mixInput, mixLab, epoch, self.modelA_ema)
                lossCNNA = lossMix.mean()
                lossCNNA_ema = lossMix_ema.mean()
                lossCNNA_mean = (lossCNNA + lossCNNA_ema)*0.5
                # update
                losses.update(lossCNNA_mean.item(), mixLab.size(0))
                precisions.update(precCNNA, mixLab.size(0))
                # update CNNA
                self.optimizers[0].zero_grad()
                lossCNNA_mean.backward()
                for param in self.modelA.parameters():
                    try:
                        param.grad.data.clamp(-1., 1.)
                    except:
                        continue
                # update modelA
                self.optimizers[0].step()
                
                self._update_ema_variables(self.modelA, self.modelA_ema, self.alpha, epoch*len(self.train_in4_data)+i)
                for param in self.modelA_ema.parameters():
                    try:
                        param.grad.data.clamp(-1., 1.)
                    except:
                        continue


            batch_time.update(time.time() - end)
            end = time.time()
            if (i + 1) % self.print_freq == 0:
                print('[AB]Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(self.train_in4_data),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              precisions.val, precisions.avg))

    def _update_ema_variables(self, model, ema_model, alpha, global_step):
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)        

    def _parse_data(self, inputs):
        imgs, fname, pids, _ = inputs
        inputs = [Variable(imgs).cuda()]
        targets = Variable(pids.cuda())
        return inputs, targets, fname  # image and pid

    def _forward(self, inputs, targets, epoch, model):
        outputs = model(*inputs)  # outputs=[x1,x2,x3]
        # x1 triplet loss
        loss_tri, _ = self.criterions[0](outputs[0], targets, epoch)  # feature
        # x2 triplet loss
        loss_global, prec_global = self.criterions[1](outputs[1], targets, epoch)
        return loss_tri + loss_global, prec_global


class CoTrainerAsyMean_6model(object):
    def saveImg(self, imgList, gtList):
        import shutil
        import os
        rootDir = self.noiseData.dataset.root
        if os.path.exists('smallLoss'):
            shutil.rmtree('smallLoss')
        os.makedirs('smallLoss')
        for name, pid in zip(imgList, gtList):
            curPath = os.path.join(rootDir, name)
            nameList = name.split('_')
            nameList[0] = str(pid)
            tarPath = os.path.join('smallLoss', '_'.join(nameList))
            shutil.copyfile(curPath, tarPath)

    def __init__(self, model, coModel, co2Model, co3Model, co4Model, co5Model, model_ema, coModel_ema, co2Model_ema, co3Model_ema,
                 co4Model_ema, co5Model_ema, train_in5_loader, train_out5_loader, train_out4_loader, train_out3_loader, train_out2_loader, unLoader,
                 criterions, optimizers, alpha=0.999, print_freq=1):
        self.modelA = model
        self.modelB = coModel  # the co-teacher
        self.modelC = co2Model
        self.modelD = co3Model
        self.modelE = co4Model
        self.modelF = co5Model
        self.modelA_ema = model_ema
        self.modelB_ema = coModel_ema
        self.modelC_ema = co2Model_ema
        self.modelD_ema = co3Model_ema
        self.modelE_ema = co4Model_ema
        self.modelF_ema = co5Model_ema
        # self.noiseData = noiseDataSet
        self.train_in5_data = train_in5_loader
        self.train_out5_data = train_out5_loader
        self.train_out4_data = train_out4_loader
        self.train_out3_data = train_out3_loader
        self.train_out2_data = train_out2_loader
        self.train_out_data = unLoader
        self.criterions = criterions
        self.optimizers = optimizers
        self.alpha = alpha
        self.print_freq = print_freq

    def train(self, epoch, remRate=0.2):
        self.modelA.train()
        self.modelB.train()
        self.modelC.train()
        self.modelD.train()
        self.modelE.train()
        self.modelF.train()
        self.modelA_ema.train()
        self.modelB_ema.train()
        self.modelC_ema.train()
        self.modelD_ema.train()
        self.modelE_ema.train()
        self.modelF_ema.train()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()
        end = time.time()

        for i, inputs in enumerate(self.train_in5_data):
            data_time.update(time.time() - end)
            inputs, targets, names = self._parse_data(inputs)  # image and pid
            # print('inputs.type: {} ,inputs.size: {} '.format(type(inputs), len(inputs)))
            # print('inputs.type: {} ,inputs[0].size: {} '.format(type(inputs), len(inputs[0])))
            # print('inputs.type: {} ,inputs[0][0].size: {} '.format(type(inputs), len(inputs[0][0])))
            # print('inputs.type: {} ,inputs[1].size: {} '.format(type(inputs), len(inputs[1])))
            # print(inputs[0])
            # print(np.array(inputs).shape)
            if i % 2 == 0:
                # update CNNF
                lossPure, prec1 = self._forward(inputs, targets, epoch, self.modelA)  # assigned samples
                # # assigned's easy samples
                lossIdx = np.argsort(lossPure.data.cpu()).cuda()
                pureInput = [inputs[0][lossIdx[:int(remRate * lossPure.shape[0])], ...]]
                pureLab = targets[lossIdx[:int(remRate * lossPure.shape[0])]].long()
                # loss for cnn F
                lossCNNF, precCNNF = self._forward(pureInput, pureLab, epoch, self.modelF)
                lossCNNF_ema, precCNNF_ema = self._forward(pureInput, pureLab, epoch, self.modelF_ema)
                lossCNNF = lossCNNF.mean()
                lossCNNF_ema = lossCNNF_ema.mean()
                lossCNNF_mean = (lossCNNF + lossCNNF_ema) * 0.5
                losses.update(lossCNNF_mean.item(), pureLab.size(0))
                precisions.update(precCNNF, pureLab.size(0))
                self.optimizers[5].zero_grad()
                lossCNNF_mean.backward()
                for param in self.modelE.parameters():
                    try:
                        param.grad.data.clamp(-1., 1.)
                    except:
                        continue
                self.optimizers[5].step()

                self._update_ema_variables(self.modelF, self.modelF_ema, self.alpha,
                                           epoch * len(self.train_in5_data) + i)
                for param in self.modelE_ema.parameters():  # 梯度截断
                    try:
                        param.grad.data.clamp(-1., 1.)
                    except:
                        continue
            else:
                # update CNNA
                try:
                    noiseInput = next(self.train_out_data)
                except:
                    noiseLoader = iter(self.train_out_data)
                    noiseInput = next(noiseLoader)
                noiseInput, noiseLab, noiseNames = self._parse_data(noiseInput)
                lossNoise, prec1 = self._forward(noiseInput, noiseLab, epoch, self.modelF)
                # sample mining
                lossIdx = np.argsort(lossNoise.data.cpu()).cuda()
                noiseInput, noiseLab = noiseInput[0][lossIdx], noiseLab[lossIdx]
                noiseInput, noiseLab = noiseInput[:int(remRate * lossNoise.shape[0]), ...], noiseLab[:int(
                    remRate * lossNoise.shape[0])]
                # save small samples
                # noiseImg = np.asarray(noiseNames)[lossIdx][:int(remRate*lossNoise.shape[0])]
                # self.saveImg(noiseImg, noiseLab) # save image
                # mix update, part assigned and part unassigned
                mixInput, mixLab = [torch.cat([inputs[0], noiseInput])], torch.cat([targets.long(), noiseLab])
                lossMix, precCNNA = self._forward(mixInput, mixLab, epoch, self.modelA)
                lossMix_ema, precCNNA_ema = self._forward(mixInput, mixLab, epoch, self.modelA_ema)
                lossCNNA = lossMix.mean()
                lossCNNA_ema = lossMix_ema.mean()
                lossCNNA_mean = (lossCNNA + lossCNNA_ema) * 0.5
                # update
                losses.update(lossCNNA_mean.item(), mixLab.size(0))
                precisions.update(precCNNA, mixLab.size(0))
                # update CNNA
                self.optimizers[0].zero_grad()
                lossCNNA_mean.backward()
                for param in self.modelA.parameters():
                    try:
                        param.grad.data.clamp(-1., 1.)
                    except:
                        continue
                # update modelA
                self.optimizers[0].step()

                self._update_ema_variables(self.modelA, self.modelA_ema, self.alpha,
                                           epoch * len(self.train_in5_data) + i)
                for param in self.modelA_ema.parameters():
                    try:
                        param.grad.data.clamp(-1., 1.)
                    except:
                        continue

            batch_time.update(time.time() - end)
            end = time.time()
            if (i + 1) % self.print_freq == 0:
                print('[AF]Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(self.train_in5_data),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              precisions.val, precisions.avg))

#================================================================================================================

        for i, inputs in enumerate(self.train_in5_data):
            data_time.update(time.time() - end)
            inputs, targets, names = self._parse_data(inputs)  # image and pid
            # print('inputs.type: {} ,inputs.size: {} '.format(type(inputs), len(inputs)))
            # print('inputs.type: {} ,inputs[0].size: {} '.format(type(inputs), len(inputs[0])))
            # print('inputs.type: {} ,inputs[0][0].size: {} '.format(type(inputs), len(inputs[0][0])))
            # print('inputs.type: {} ,inputs[1].size: {} '.format(type(inputs), len(inputs[1])))
            # print(inputs[0])
            # print(np.array(inputs).shape)
            if i % 2 == 0:
                # update CNNE
                lossPure, prec1 = self._forward(inputs, targets, epoch, self.modelA)  # assigned samples
                # # assigned's easy samples
                lossIdx = np.argsort(lossPure.data.cpu()).cuda()
                pureInput = [inputs[0][lossIdx[:int(remRate * lossPure.shape[0])], ...]]
                pureLab = targets[lossIdx[:int(remRate * lossPure.shape[0])]].long()
                # loss for cnn D
                lossCNNE, precCNNE = self._forward(pureInput, pureLab, epoch, self.modelE)
                lossCNNE_ema, precCNNE_ema = self._forward(pureInput, pureLab, epoch, self.modelE_ema)
                lossCNNE = lossCNNE.mean()
                lossCNNE_ema = lossCNNE_ema.mean()
                lossCNNE_mean = (lossCNNE + lossCNNE_ema) * 0.5
                losses.update(lossCNNE_mean.item(), pureLab.size(0))
                precisions.update(precCNNE, pureLab.size(0))
                self.optimizers[4].zero_grad()
                lossCNNE_mean.backward()
                for param in self.modelE.parameters():
                    try:
                        param.grad.data.clamp(-1., 1.)
                    except:
                        continue
                self.optimizers[4].step()

                self._update_ema_variables(self.modelE, self.modelE_ema, self.alpha,
                                           epoch * len(self.train_in5_data) + i)
                for param in self.modelE_ema.parameters():  # 梯度截断
                    try:
                        param.grad.data.clamp(-1., 1.)
                    except:
                        continue
            else:
                # update CNNA
                try:
                    noiseInput = next(self.train_out_data)
                except:
                    noiseLoader = iter(self.train_out_data)
                    noiseInput = next(noiseLoader)
                noiseInput, noiseLab, noiseNames = self._parse_data(noiseInput)
                lossNoise, prec1 = self._forward(noiseInput, noiseLab, epoch, self.modelE)
                # sample mining
                lossIdx = np.argsort(lossNoise.data.cpu()).cuda()
                noiseInput, noiseLab = noiseInput[0][lossIdx], noiseLab[lossIdx]
                noiseInput, noiseLab = noiseInput[:int(remRate * lossNoise.shape[0]), ...], noiseLab[:int(
                    remRate * lossNoise.shape[0])]
                # save small samples
                # noiseImg = np.asarray(noiseNames)[lossIdx][:int(remRate*lossNoise.shape[0])]
                # self.saveImg(noiseImg, noiseLab) # save image
                # mix update, part assigned and part unassigned
                mixInput, mixLab = [torch.cat([inputs[0], noiseInput])], torch.cat([targets.long(), noiseLab])
                lossMix, precCNNA = self._forward(mixInput, mixLab, epoch, self.modelA)
                lossMix_ema, precCNNA_ema = self._forward(mixInput, mixLab, epoch, self.modelA_ema)
                lossCNNA = lossMix.mean()
                lossCNNA_ema = lossMix_ema.mean()
                lossCNNA_mean = (lossCNNA + lossCNNA_ema) * 0.5
                # update
                losses.update(lossCNNA_mean.item(), mixLab.size(0))
                precisions.update(precCNNA, mixLab.size(0))
                # update CNNA
                self.optimizers[0].zero_grad()
                lossCNNA_mean.backward()
                for param in self.modelA.parameters():
                    try:
                        param.grad.data.clamp(-1., 1.)
                    except:
                        continue
                # update modelA
                self.optimizers[0].step()

                self._update_ema_variables(self.modelA, self.modelA_ema, self.alpha,
                                           epoch * len(self.train_in5_data) + i)
                for param in self.modelA_ema.parameters():
                    try:
                        param.grad.data.clamp(-1., 1.)
                    except:
                        continue

            batch_time.update(time.time() - end)
            end = time.time()
            if (i + 1) % self.print_freq == 0:
                print('[AE]Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(self.train_in5_data),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              precisions.val, precisions.avg))

        for i, inputs in enumerate(self.train_in5_data):
            data_time.update(time.time() - end)
            inputs, targets, names = self._parse_data(inputs)  # image and pid
            # print('inputs.type: {} ,inputs.size: {} '.format(type(inputs), len(inputs)))
            # print('inputs.type: {} ,inputs[0].size: {} '.format(type(inputs), len(inputs[0])))
            # print('inputs.type: {} ,inputs[0][0].size: {} '.format(type(inputs), len(inputs[0][0])))
            # print('inputs.type: {} ,inputs[1].size: {} '.format(type(inputs), len(inputs[1])))
            # print(inputs[0])
            # print(np.array(inputs).shape)
            if i % 2 == 0:
                # update CNND
                lossPure, prec1 = self._forward(inputs, targets, epoch, self.modelA)  # assigned samples
                # # assigned's easy samples
                lossIdx = np.argsort(lossPure.data.cpu()).cuda()
                pureInput = [inputs[0][lossIdx[:int(remRate * lossPure.shape[0])], ...]]
                pureLab = targets[lossIdx[:int(remRate * lossPure.shape[0])]].long()
                # loss for cnn D
                lossCNND, precCNND = self._forward(pureInput, pureLab, epoch, self.modelD)
                lossCNND_ema, precCNND_ema = self._forward(pureInput, pureLab, epoch, self.modelD_ema)
                lossCNND = lossCNND.mean()
                lossCNND_ema = lossCNND_ema.mean()
                lossCNND_mean = (lossCNND + lossCNND_ema) * 0.5
                losses.update(lossCNND_mean.item(), pureLab.size(0))
                precisions.update(precCNND, pureLab.size(0))
                self.optimizers[3].zero_grad()
                lossCNND_mean.backward()
                for param in self.modelD.parameters():
                    try:
                        param.grad.data.clamp(-1., 1.)
                    except:
                        continue
                self.optimizers[3].step()

                self._update_ema_variables(self.modelD, self.modelD_ema, self.alpha,
                                           epoch * len(self.train_in5_data) + i)
                for param in self.modelD_ema.parameters():  # 梯度截断
                    try:
                        param.grad.data.clamp(-1., 1.)
                    except:
                        continue
            else:
                # update CNNA
                try:
                    noiseInput = next(self.train_out2_data)
                except:
                    noiseLoader = iter(self.train_out2_data)
                    noiseInput = next(noiseLoader)
                noiseInput, noiseLab, noiseNames = self._parse_data(noiseInput)
                lossNoise, prec1 = self._forward(noiseInput, noiseLab, epoch, self.modelD)
                # sample mining
                lossIdx = np.argsort(lossNoise.data.cpu()).cuda()
                noiseInput, noiseLab = noiseInput[0][lossIdx], noiseLab[lossIdx]
                noiseInput, noiseLab = noiseInput[:int(remRate * lossNoise.shape[0]), ...], noiseLab[:int(
                    remRate * lossNoise.shape[0])]
                # save small samples
                # noiseImg = np.asarray(noiseNames)[lossIdx][:int(remRate*lossNoise.shape[0])]
                # self.saveImg(noiseImg, noiseLab) # save image
                # mix update, part assigned and part unassigned
                mixInput, mixLab = [torch.cat([inputs[0], noiseInput])], torch.cat([targets.long(), noiseLab])
                lossMix, precCNNA = self._forward(mixInput, mixLab, epoch, self.modelA)
                lossMix_ema, precCNNA_ema = self._forward(mixInput, mixLab, epoch, self.modelA_ema)
                lossCNNA = lossMix.mean()
                lossCNNA_ema = lossMix_ema.mean()
                lossCNNA_mean = (lossCNNA + lossCNNA_ema) * 0.5
                # update
                losses.update(lossCNNA_mean.item(), mixLab.size(0))
                precisions.update(precCNNA, mixLab.size(0))
                # update CNNA
                self.optimizers[0].zero_grad()
                lossCNNA_mean.backward()
                for param in self.modelA.parameters():
                    try:
                        param.grad.data.clamp(-1., 1.)
                    except:
                        continue
                # update modelA
                self.optimizers[0].step()

                self._update_ema_variables(self.modelA, self.modelA_ema, self.alpha,
                                           epoch * len(self.train_in5_data) + i)
                for param in self.modelA_ema.parameters():
                    try:
                        param.grad.data.clamp(-1., 1.)
                    except:
                        continue

            batch_time.update(time.time() - end)
            end = time.time()
            if (i + 1) % self.print_freq == 0:
                print('[AD]Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(self.train_in5_data),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              precisions.val, precisions.avg))

        for i, inputs in enumerate(self.train_in5_data):
            data_time.update(time.time() - end)
            inputs, targets, names = self._parse_data(inputs)  # image and pid
            # print('inputs.type: {} ,inputs.size: {} '.format(type(inputs), len(inputs)))
            # print('inputs.type: {} ,inputs[0].size: {} '.format(type(inputs), len(inputs[0])))
            # print('inputs.type: {} ,inputs[0][0].size: {} '.format(type(inputs), len(inputs[0][0])))
            # print('inputs.type: {} ,inputs[1].size: {} '.format(type(inputs), len(inputs[1])))
            # print(inputs[0])
            # print(np.array(inputs).shape)
            if i % 2 == 0:
                # update CNNC
                lossPure, prec1 = self._forward(inputs, targets, epoch, self.modelA)  # assigned samples
                # # assigned's easy samples
                lossIdx = np.argsort(lossPure.data.cpu()).cuda()
                pureInput = [inputs[0][lossIdx[:int(remRate * lossPure.shape[0])], ...]]
                pureLab = targets[lossIdx[:int(remRate * lossPure.shape[0])]].long()
                # loss for cnn C
                lossCNNC, precCNNC = self._forward(pureInput, pureLab, epoch, self.modelC)
                lossCNNC_ema, precCNNC_ema = self._forward(pureInput, pureLab, epoch, self.modelC_ema)
                lossCNNC = lossCNNC.mean()
                lossCNNC_ema = lossCNNC_ema.mean()
                lossCNNC_mean = (lossCNNC + lossCNNC_ema) * 0.5
                losses.update(lossCNNC_mean.item(), pureLab.size(0))
                precisions.update(precCNNC, pureLab.size(0))
                self.optimizers[2].zero_grad()
                lossCNNC_mean.backward()
                for param in self.modelC.parameters():
                    try:
                        param.grad.data.clamp(-1., 1.)
                    except:
                        continue
                self.optimizers[2].step()

                self._update_ema_variables(self.modelC, self.modelC_ema, self.alpha,
                                           epoch * len(self.train_in5_data) + i)
                for param in self.modelC_ema.parameters():  # 梯度截断
                    try:
                        param.grad.data.clamp(-1., 1.)
                    except:
                        continue

            else:
                # update CNNA
                try:
                    noiseInput = next(self.train_out3_data)
                except:
                    noiseLoader = iter(self.train_out3_data)
                    noiseInput = next(noiseLoader)
                noiseInput, noiseLab, noiseNames = self._parse_data(noiseInput)
                lossNoise, prec1 = self._forward(noiseInput, noiseLab, epoch, self.modelC)
                # sample mining
                lossIdx = np.argsort(lossNoise.data.cpu()).cuda()
                noiseInput, noiseLab = noiseInput[0][lossIdx], noiseLab[lossIdx]
                noiseInput, noiseLab = noiseInput[:int(remRate * lossNoise.shape[0]), ...], noiseLab[:int(
                    remRate * lossNoise.shape[0])]
                # save small samples
                # noiseImg = np.asarray(noiseNames)[lossIdx][:int(remRate*lossNoise.shape[0])]
                # self.saveImg(noiseImg, noiseLab) # save image
                # mix update, part assigned and part unassigned
                mixInput, mixLab = [torch.cat([inputs[0], noiseInput])], torch.cat([targets.long(), noiseLab])
                lossMix, precCNNA = self._forward(mixInput, mixLab, epoch, self.modelA)
                lossMix_ema, precCNNA_ema = self._forward(mixInput, mixLab, epoch, self.modelA_ema)
                lossCNNA = lossMix.mean()
                lossCNNA_ema = lossMix_ema.mean()
                lossCNNA_mean = (lossCNNA + lossCNNA_ema) * 0.5
                # update
                losses.update(lossCNNA_mean.item(), mixLab.size(0))
                precisions.update(precCNNA, mixLab.size(0))
                # update CNNA
                self.optimizers[0].zero_grad()
                lossCNNA_mean.backward()
                for param in self.modelA.parameters():
                    try:
                        param.grad.data.clamp(-1., 1.)
                    except:
                        continue
                # update modelA
                self.optimizers[0].step()

                self._update_ema_variables(self.modelA, self.modelA_ema, self.alpha,
                                           epoch * len(self.train_in5_data) + i)
                for param in self.modelA_ema.parameters():
                    try:
                        param.grad.data.clamp(-1., 1.)
                    except:
                        continue

            batch_time.update(time.time() - end)
            end = time.time()
            if (i + 1) % self.print_freq == 0:
                print('[AC]Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(self.train_in5_data),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              precisions.val, precisions.avg))

        for i, inputs in enumerate(self.train_in5_data):
            data_time.update(time.time() - end)
            inputs, targets, names = self._parse_data(inputs)  # image and pid
            # print('inputs.type: {} ,inputs.size: {} '.format(type(inputs), len(inputs)))
            # print('inputs.type: {} ,inputs[0].size: {} '.format(type(inputs), len(inputs[0])))
            # print('inputs.type: {} ,inputs[0][0].size: {} '.format(type(inputs), len(inputs[0][0])))
            # print('inputs.type: {} ,inputs[1].size: {} '.format(type(inputs), len(inputs[1])))
            # print(inputs[0])
            # print(np.array(inputs).shape)
            if i % 2 == 0:
                # update CNNB
                lossPure, prec1 = self._forward(inputs, targets, epoch, self.modelA)  # assigned samples
                # # assigned's easy samples
                lossIdx = np.argsort(lossPure.data.cpu()).cuda()
                pureInput = [inputs[0][lossIdx[:int(remRate * lossPure.shape[0])], ...]]
                pureLab = targets[lossIdx[:int(remRate * lossPure.shape[0])]].long()
                # loss for cnn B
                lossCNNB, precCNNB = self._forward(pureInput, pureLab, epoch, self.modelB)
                lossCNNB_ema, precCNNB_ema = self._forward(pureInput, pureLab, epoch, self.modelB_ema)
                lossCNNB = lossCNNB.mean()
                lossCNNB_ema = lossCNNB_ema.mean()
                lossCNNB_mean = (lossCNNB + lossCNNB_ema) * 0.5
                losses.update(lossCNNB_mean.item(), pureLab.size(0))
                precisions.update(precCNNB, pureLab.size(0))
                self.optimizers[1].zero_grad()
                lossCNNB_mean.backward()
                for param in self.modelB.parameters():
                    try:
                        param.grad.data.clamp(-1., 1.)
                    except:
                        continue
                self.optimizers[1].step()

                self._update_ema_variables(self.modelB, self.modelB_ema, self.alpha,
                                           epoch * len(self.train_in5_data) + i)
                for param in self.modelB_ema.parameters():  # 梯度截断
                    try:
                        param.grad.data.clamp(-1., 1.)
                    except:
                        continue
            else:
                # update CNNA
                try:
                    noiseInput = next(self.train_out4_data)
                except:
                    noiseLoader = iter(self.train_out4_data)
                    noiseInput = next(noiseLoader)
                noiseInput, noiseLab, noiseNames = self._parse_data(noiseInput)
                lossNoise, prec1 = self._forward(noiseInput, noiseLab, epoch, self.modelB)
                # sample mining
                lossIdx = np.argsort(lossNoise.data.cpu()).cuda()
                noiseInput, noiseLab = noiseInput[0][lossIdx], noiseLab[lossIdx]
                noiseInput, noiseLab = noiseInput[:int(remRate * lossNoise.shape[0]), ...], noiseLab[:int(
                    remRate * lossNoise.shape[0])]
                # save small samples
                # noiseImg = np.asarray(noiseNames)[lossIdx][:int(remRate*lossNoise.shape[0])]
                # self.saveImg(noiseImg, noiseLab) # save image
                # mix update, part assigned and part unassigned
                mixInput, mixLab = [torch.cat([inputs[0], noiseInput])], torch.cat([targets.long(), noiseLab])
                lossMix, precCNNA = self._forward(mixInput, mixLab, epoch, self.modelA)
                lossMix_ema, precCNNA_ema = self._forward(mixInput, mixLab, epoch, self.modelA_ema)
                lossCNNA = lossMix.mean()
                lossCNNA_ema = lossMix_ema.mean()
                lossCNNA_mean = (lossCNNA + lossCNNA_ema) * 0.5
                # update
                losses.update(lossCNNA_mean.item(), mixLab.size(0))
                precisions.update(precCNNA, mixLab.size(0))
                # update CNNA
                self.optimizers[0].zero_grad()
                lossCNNA_mean.backward()
                for param in self.modelA.parameters():
                    try:
                        param.grad.data.clamp(-1., 1.)
                    except:
                        continue
                # update modelA
                self.optimizers[0].step()

                self._update_ema_variables(self.modelA, self.modelA_ema, self.alpha,
                                           epoch * len(self.train_in5_data) + i)
                for param in self.modelA_ema.parameters():
                    try:
                        param.grad.data.clamp(-1., 1.)
                    except:
                        continue

            batch_time.update(time.time() - end)
            end = time.time()
            if (i + 1) % self.print_freq == 0:
                print('[AB]Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(self.train_in5_data),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              precisions.val, precisions.avg))

    def _update_ema_variables(self, model, ema_model, alpha, global_step):
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    def _parse_data(self, inputs):
        imgs, fname, pids, _ = inputs
        inputs = [Variable(imgs).cuda()]
        targets = Variable(pids.cuda())
        return inputs, targets, fname  # image and pid

    def _forward(self, inputs, targets, epoch, model):
        outputs = model(*inputs)  # outputs=[x1,x2,x3]
        # x1 triplet loss
        loss_tri, _ = self.criterions[0](outputs[0], targets, epoch)  # feature
        # x2 triplet loss
        loss_global, prec_global = self.criterions[1](outputs[1], targets, epoch)
        return loss_tri + loss_global, prec_global




class CoTrainerAsyMean2(object):#α-liner
    def saveImg(self, imgList, gtList):
        import shutil
        import os
        rootDir = self.noiseData.dataset.root
        if os.path.exists('smallLoss'):
            shutil.rmtree('smallLoss')
        os.makedirs('smallLoss')
        for name, pid in zip(imgList, gtList):
            curPath = os.path.join(rootDir, name)
            nameList = name.split('_')
            nameList[0] = str(pid)
            tarPath = os.path.join('smallLoss', '_'.join(nameList))
            shutil.copyfile(curPath, tarPath)

    def __init__(self, model, coModel, model_ema, coModel_ema, newDataSet, noiseDataSet, criterions, optimizers, alpha=0.999, print_freq=1):
        self.modelA = model
        self.modelB = coModel  # the co-teacher
        self.modelA_ema = model_ema
        self.modelB_ema = coModel_ema
        self.noiseData = noiseDataSet
        self.newDataSet = newDataSet
        self.criterions = criterions
        self.optimizers = optimizers
        self.alpha = alpha
        self.print_freq = print_freq

    def train(self, epoch, remRate=0.2):
        self.modelA.train()
        self.modelB.train()
        self.modelA_ema.train()
        self.modelB_ema.train()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()
        #precisions = [AverageMeter(),AverageMeter()]
        end = time.time()

        for i, inputs in enumerate(self.newDataSet):
            data_time.update(time.time() - end)
            inputs, targets, names = self._parse_data(inputs)  # image and pid
            
            self.alpha = self.alpha -(self.alpha-0.5)* (i+1)/len(self.newDataSet)
            
            
            
            if i % 2 == 0:
                # update CNNB
                lossPure, prec1 = self._forward(inputs, targets, epoch, self.modelA)  # assigned samples
                #lossPure_ema, prec1_ema = self._forward(inputs, targets, epoch, self.modelA_ema)
                # # assigned's easy samples
                lossIdx = np.argsort(lossPure.data.cpu()).cuda()
                #lossIdx_ema = np.argsort(lossPure_ema.data.cpu()).cuda()
                pureInput = [inputs[0][lossIdx[:int(remRate * lossPure.shape[0])], ...]]
                #pureInput_ema = [inputs[0][lossIdx_ema[:int(remRate * lossPure_ema.shape[0])], ...]]
                pureLab = targets[lossIdx[:int(remRate * lossPure.shape[0])]].long()
                #pureLab_ema = targets[lossIdx_ema[:int(remRate * lossPure_ema.shape[0])]].long()
                # loss for cnn B
                lossCNNB, precCNNB = self._forward(pureInput, pureLab, epoch, self.modelB)
                lossCNNB_ema, precCNNB_ema = self._forward(pureInput, pureLab, epoch, self.modelB_ema)
                lossCNNB = lossCNNB.mean()
                lossCNNB_ema = lossCNNB_ema.mean()
                lossCNNB_mean = (lossCNNB + lossCNNB_ema)*0.5
                losses.update(lossCNNB_mean.item(), pureLab.size(0))
                precisions.update(precCNNB, pureLab.size(0))
                self.optimizers[1].zero_grad()
                lossCNNB_mean.backward()
                for param in self.modelB.parameters():#梯度截断
                    try:
                        param.grad.data.clamp(-1., 1.)
                    except:
                        continue
                self.optimizers[1].step()
                
                self._update_ema_variables(self.modelB, self.modelB_ema, self.alpha, epoch*len(self.newDataSet)+i)
                for param in self.modelB_ema.parameters():#梯度截断
                    try:
                        param.grad.data.clamp(-1., 1.)
                    except:
                        continue
                
            else:
                # update CNNA
                try:
                    noiseInput = next(self.noiseData)
                except:
                    noiseLoader = iter(self.noiseData)
                    noiseInput = next(noiseLoader)
                noiseInput, noiseLab, noiseNames = self._parse_data(noiseInput)
                lossNoise, prec1 = self._forward(noiseInput, noiseLab, epoch, self.modelB)
                #lossNoise_ema, prec1_ema = self._forward(noiseInput, noiseLab, epoch, self.modelB_ema)
                # sample mining
                lossIdx = np.argsort(lossNoise.data.cpu()).cuda()
                noiseInput, noiseLab = noiseInput[0][lossIdx], noiseLab[lossIdx]
                noiseInput, noiseLab = noiseInput[:int(remRate * lossNoise.shape[0]), ...], noiseLab[:int(
                    remRate * lossNoise.shape[0])]
                # save small samples
                # noiseImg = np.asarray(noiseNames)[lossIdx][:int(remRate*lossNoise.shape[0])]
                # self.saveImg(noiseImg, noiseLab) # save image
                # mix update, part assigned and part unassigned
                mixInput, mixLab = [torch.cat([inputs[0], noiseInput])], torch.cat([targets.long(), noiseLab])
                lossMix, precCNNA = self._forward(mixInput, mixLab, epoch, self.modelA)
                lossMix_ema, precCNNA_ema = self._forward(mixInput, mixLab, epoch, self.modelA_ema)
                lossCNNA = lossMix.mean()
                lossCNNA_ema = lossMix_ema.mean()
                lossCNNA_mean = (lossCNNA + lossCNNA_ema)*0.5
                # update
                losses.update(lossCNNA_mean.item(), mixLab.size(0))
                precisions.update(precCNNA, mixLab.size(0))
                # update CNNA
                self.optimizers[0].zero_grad()
                lossCNNA_mean.backward()
                for param in self.modelA.parameters():
                    try:
                        param.grad.data.clamp(-1., 1.)
                    except:
                        continue
                # update modelA
                self.optimizers[0].step()
                
                self._update_ema_variables(self.modelA, self.modelA_ema, self.alpha, epoch*len(self.newDataSet)+i)
                for param in self.modelA_ema.parameters():
                    try:
                        param.grad.data.clamp(-1., 1.)
                    except:
                        continue

            batch_time.update(time.time() - end)
            end = time.time()
            if (i + 1) % self.print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(self.newDataSet),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              precisions.val, precisions.avg))
    
    def _update_ema_variables(self, model, ema_model, alpha, global_step):
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
        
    def _parse_data(self, inputs):
        imgs, fname, pids, _ = inputs
        inputs = [Variable(imgs).cuda()]
        targets = Variable(pids.cuda())
        return inputs, targets, fname  # image and pid

    def _forward(self, inputs, targets, epoch, model):
        outputs = model(*inputs)  # outputs=[x1,x2,x3]
        # x1 triplet loss
        loss_tri, _ = self.criterions[0](outputs[0], targets, epoch)  # feature
        # x2 triplet loss
        loss_global, prec_global = self.criterions[1](outputs[1], targets, epoch)
        return loss_tri + loss_global, prec_global


class CoTrainerAsySep(object):
    def __init__(self, model, coModel, newDataSet, noiseDataSet, criterions, optimizers, print_freq=1):
        self.modelA = model
        self.modelB = coModel  # the co-teacher
        self.noiseData = noiseDataSet
        self.newDataSet = newDataSet
        self.criterions = criterions
        self.optimizers = optimizers
        self.print_freq = print_freq

    def train(self, epoch, remRate=0.2):
        self.modelA.train()
        self.modelB.train()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()
        end = time.time()

        for i, inputs in enumerate(self.newDataSet):
            data_time.update(time.time() - end)
            inputs, targets = self._parse_data(inputs)  # image and pid
            if i % 2 == 0:
                # update CNNB
                lossPure, prec1 = self._forward(inputs, targets, epoch, self.modelA)  # assigned samples
                # # assigned's easy samples
                lossIdx = np.argsort(lossPure.data.cpu()).cuda()
                pureInput = [inputs[0][lossIdx[:int(remRate * lossPure.shape[0])], ...]]
                pureLab = targets[lossIdx[:int(remRate * lossPure.shape[0])]]
                # loss for cnn B
                lossCNNB, precCNNB = self._forward(pureInput, pureLab, epoch, self.modelB)
                lossCNNB = lossCNNB.mean()
                losses.update(lossCNNB.item(), pureLab.size(0))
                precisions.update(precCNNB, pureLab.size(0))
                self.optimizers[1].zero_grad()
                lossCNNB.backward()
                for param in self.modelB.parameters():
                    try:
                        param.grad.data.clamp(-1., 1.)
                    except:
                        continue
                self.optimizers[1].step()
            else:
                # update CNNA
                try:
                    noiseInput = next(self.noiseData)
                except:
                    noiseLoader = iter(self.noiseData)
                    noiseInput = next(noiseLoader)
                noiseInput, noiseLab = self._parse_data(noiseInput)
                lossNoise, prec1 = self._forward(noiseInput, noiseLab, epoch, self.modelB)
                # sample mining
                lossIdx = np.argsort(lossNoise.data.cpu()).cuda()
                noiseInput, noiseLab = noiseInput[0][lossIdx], noiseLab[lossIdx]
                noiseInput, noiseLab = noiseInput[:int(remRate * lossNoise.shape[0]), ...], noiseLab[:int(
                    remRate * lossNoise.shape[0])]
                # mix update, part assigned and part unassigned
                # mixInput, mixLab = [torch.cat([inputs[0],noiseInput])], torch.cat([targets,noiseLab])
                lossCNNAnoise, precCNNAnoise = self._forward([noiseInput], noiseLab, epoch, self.modelA)
                lossCNNApure, precCNNApure = self._forward(inputs, targets, epoch, self.modelA)
                lossCNNA = 0.1 * lossCNNAnoise.mean() + lossCNNApure.mean()
                # update
                losses.update(lossCNNA.item(), targets.size(0))
                precisions.update(precCNNApure, targets.size(0))
                # update CNNA
                self.optimizers[0].zero_grad()
                lossCNNA.backward()
                for param in self.modelA.parameters():
                    try:
                        param.grad.data.clamp(-1., 1.)
                    except:
                        continue
                # update modelA
                self.optimizers[0].step()

            batch_time.update(time.time() - end)
            end = time.time()
            if (i + 1) % self.print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(self.newDataSet),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              precisions.val, precisions.avg))

    def _parse_data(self, inputs):
        imgs, fname, pids, _ = inputs
        inputs = [Variable(imgs).cuda()]
        targets = Variable(pids.cuda())
        return inputs, targets  # image and pid

    def _forward(self, inputs, targets, epoch, model):
        outputs = model(*inputs)  # outputs=[x1,x2,x3]
        # x1 triplet loss
        loss_tri, _ = self.criterions[0](outputs[0], targets, epoch)  # feature
        # x2 triplet loss
        loss_global, prec_global = self.criterions[1](outputs[1], targets, epoch)
        return loss_tri + loss_global, prec_global


class EvoTrainer(object):
    def __init__(self, model, newDataSet, noiseDataSet, criterions, optimizer, print_freq=1):
        self.model = model
        self.noiseData = noiseDataSet
        self.newDataSet = newDataSet
        self.criterions = criterions
        self.optimizer = optimizer
        self.print_freq = print_freq

    def train(self, epoch, remRate=0.2):
        self.model.train()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()
        end = time.time()

        for i, inputs in enumerate(self.newDataSet):
            data_time.update(time.time() - end)
            inputs, targets = self._parse_data(inputs)  # image and pid
            # update CNNA
            lossPure, prec1 = self._forward(inputs, targets, epoch, self.model)  # assigned samples
            pureIdx = np.argsort(lossPure.data.cpu()).cuda()
            pureInput, targets = inputs[0][pureIdx], targets[pureIdx]
            pureInput, targets = pureInput[:int(remRate * lossPure.shape[0]), ...], targets[
                                                                                    :int(remRate * lossPure.shape[0])]
            # update CNNA
            try:
                noiseInput = next(noiseLoader)
            except:
                noiseLoader = iter(self.noiseData)
                noiseInput = next(noiseLoader)
            noiseInput, noiseLab = self._parse_data(noiseInput)
            lossNoise, prec1 = self._forward(noiseInput, noiseLab, epoch, self.model)
            # sample mining
            lossIdx = np.argsort(lossNoise.data.cpu()).cuda()
            noiseInput, noiseLab = noiseInput[0][lossIdx], noiseLab[lossIdx]
            noiseInput, noiseLab = noiseInput[:int(remRate * lossNoise.shape[0]), ...], noiseLab[:int(
                remRate * lossNoise.shape[0])]
            # mix update, part assigned and part unassigned
            mixInput, mixLab = [torch.cat([pureInput, noiseInput])], torch.cat([targets, noiseLab])
            lossMix, precCNNA = self._forward(mixInput, mixLab, epoch, self.model)
            lossCNNA = lossMix.mean()
            # update
            losses.update(lossCNNA.item(), mixLab.size(0))
            precisions.update(precCNNA, mixLab.size(0))
            # update CNNA
            self.optimizer.zero_grad()
            lossCNNA.backward()
            for param in self.model.parameters():
                try:
                    param.grad.data.clamp(-1., 1.)
                except:
                    continue
            # update modelA
            self.optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()
            if (i + 1) % self.print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(self.newDataSet),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              precisions.val, precisions.avg))

    def _parse_data(self, inputs):
        imgs, fname, pids, _ = inputs
        inputs = [Variable(imgs).cuda()]
        targets = Variable(pids.cuda())
        return inputs, targets  # image and pid

    def _forward(self, inputs, targets, epoch, model):
        outputs = model(*inputs)  # outputs=[x1,x2,x3]
        # x1 triplet loss
        loss_tri, _ = self.criterions[0](outputs[0], targets, epoch)  # feature
        # x2 triplet loss
        loss_global, prec_global = self.criterions[1](outputs[1], targets, epoch)
        return loss_tri + loss_global, prec_global
