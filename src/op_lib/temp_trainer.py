import torch
from torch import nn
import torchvision
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, PolynomialLR
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import numpy as np
import time

from .hdf5_dataset import HDF5Dataset, TempVelDataset
from .metrics import compute_metrics, write_metrics
from .losses import LpLoss
from .plt_util import plt_temp, plt_iter_mae
from .heatflux import heatflux

from torch.cuda import nvtx 

t_bulk_map = {
    'wall_super_heat': 58,
    'subcooled': 50
}

class TempTrainer:
    def __init__(self,
                 model,
                 train_dataloader,
                 val_dataloader,
                 optimizer,
                 lr_scheduler,
                 val_variable,
                 writer,
                 cfg):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.val_variable = val_variable
        self.writer = writer
        self.cfg = cfg
        self.loss = LpLoss(d=2)

    def train(self, max_epochs):
        for epoch in range(max_epochs):
            print('epoch ', epoch)
            self.train_step(epoch)
            self.val_step(epoch)
            self.lr_scheduler.step()

    def train_step(self, epoch):
        self.model.train()
        for iter, (input, label) in enumerate(self.train_dataloader):
            input = input.cuda().float()
            label = label.cuda().float()
            
            pred = self.model(input)
            print(pred.size(), label.size())
            temp_loss = self.loss(pred, label)
            loss = temp_loss
            torch.cuda.synchronize()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            torch.cuda.synchronize()
            
            print(f'train loss: {loss}')

            global_iter = epoch * len(self.train_dataloader) + iter
            write_metrics(pred, label, global_iter, 'Train', self.writer)

            del input, label

    def val_step(self, epoch):
        self.model.eval()
        for iter, (input, label) in enumerate(self.val_dataloader):
            input = input.cuda().float()
            label = label.cuda().float()
            with torch.no_grad():
                pred = self.model(input)
                temp_loss = F.mse_loss(pred, label)
                loss = temp_loss
            print(f'val loss: {loss}')
            global_iter = epoch * len(self.val_dataloader) + iter
            write_metrics(pred, label, global_iter, 'Val', self.writer)
            del input, label

    def test(self, dataset):
        self.model.eval()
        temps = []
        labels = []
        inference_times = []
        for timestep in range(len(dataset)):
            start = time.perf_counter()
            input, label = dataset[timestep]
            label = label.cuda().float().unsqueeze(0)
            input = input.cuda().float().unsqueeze(0)
            with torch.no_grad():
                pred = self.model(input)
                temp = F.hardtanh(pred[:, 0], min_val=0, max_val=1)
                inference_times.append(time.perf_counter()-start)
                dataset.write_temp(temp, timestep)
                temps.append(temp.detach().cpu())
                labels.append(label[:, 0].detach().cpu())
        temps = torch.cat(temps, dim=0)
        labels = torch.cat(labels, dim=0)
        
        print(f'Total inference time for {len(dataset)} frames: {sum(inference_times)}')
        dfun = dataset.get_dfun().permute((2, 0, 1))
        metrics = compute_metrics(temps, labels, dfun)
        print(metrics)
        
        xgrid = dataset.get_x().permute((2, 0, 1))
        print(heatflux(temps, dfun, self.val_variable, xgrid, dataset.get_dy()))
        print(heatflux(labels, dfun, self.val_variable, xgrid, dataset.get_dy()))
        
        plt_iter_mae(temps, labels)
        plt_temp(temps, labels, self.model.__class__.__name__)
