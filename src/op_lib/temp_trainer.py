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
                 future_window,
                 push_forward_steps,
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

        self.push_forward_steps = push_forward_steps
        self.future_window = future_window

    def train(self, max_epochs):
        for epoch in range(max_epochs):
            print('epoch ', epoch)
            self.train_step(epoch)
            self.val_step(epoch)
            self.lr_scheduler.step()

    def _forward_int(self, temp, vel):
        input = torch.cat((temp, vel), dim=1)
        pred = self.model(input)
        # TODO: account for different timesteps of training data
        #  - u_k+l = u_k + (t_k+l - t_k)d_l
        # at the moment, time discretization is 1, so t_k+l - t_k is just l + 1
        timesteps = (torch.arange(self.future_window) + 1).cuda().unsqueeze(-1).unsqueeze(-1)
        last_temp_input = temp[:, -1].unsqueeze(1)
        sol = last_temp_input + timesteps * pred
        return sol

    def push_forward_trick(self, temp, vel):
        #with torch.no_grad():
        #    for idx in range(self.push_forward_steps - 1):
        #        pred = self._forward_int(temp, vel)
        #        temp = torch.roll(temp, -self.future_window, dims=1)
        pred = self._forward_int(temp, vel)
        return pred

    def train_step(self, epoch):
        self.model.train()

        for iter, (coords, temp, vel, label) in enumerate(self.train_dataloader):
            temp = temp.cuda().float()
            vel = vel.cuda().float()
            label = label.cuda().float()
            
            pred = self.push_forward_trick(temp, vel)
            loss = F.mse_loss(pred, label)
            #loss = self.loss(pred, label).mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            print(f'train loss: {loss}')
            global_iter = epoch * len(self.train_dataloader) + iter
            write_metrics(pred, label, global_iter, 'Train', self.writer)
            del temp, vel, label

    def val_step(self, epoch):
        self.model.eval()
        for iter, (coords, temp, vel, label) in enumerate(self.val_dataloader):
            temp = temp.cuda().float()
            vel = vel.cuda().float()
            label = label.cuda().float()
            with torch.no_grad():
                pred = self._forward_int(temp, vel)
                temp_loss = F.mse_loss(pred, label)
                loss = temp_loss
            print(f'val loss: {loss}')
            global_iter = epoch * len(self.val_dataloader) + iter
            write_metrics(pred, label, global_iter, 'Val', self.writer)
            del temp, vel, label

    def test(self, dataset):
        self.model.eval()
        temps = []
        labels = []
        for timestep in range(0, len(dataset), self.future_window):
            coords, temp, vel, label = dataset[timestep]
            temp = temp.cuda().float().unsqueeze(0)
            vel = vel.cuda().float().unsqueeze(0)
            label = label.cuda().float()
            with torch.no_grad():
                pred = self._forward_int(temp, vel)
                temp = F.hardtanh(pred, min_val=-1, max_val=1).squeeze(0)
                print(pred.size(), temp.size())
                dataset.write_temp(temp.permute((1, 2, 0)), timestep)
                temps.append(temp.detach().cpu())
                labels.append(label.detach().cpu())

        temps = torch.cat(temps, dim=0)
        labels = torch.cat(labels, dim=0)
        dfun = dataset.get_dfun().permute((2, 0, 1))

        metrics = compute_metrics(temps, labels, dfun)
        print(metrics)
        
        #xgrid = dataset.get_x().permute((2, 0, 1))
        #print(heatflux(temps, dfun, self.val_variable, xgrid, dataset.get_dy()))
        #print(heatflux(labels, dfun, self.val_variable, xgrid, dataset.get_dy()))
        
        plt_iter_mae(temps, labels)
        plt_temp(temps, labels, self.model.__class__.__name__)
