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
from .metrics import compute_metrics
from .losses import LpLoss
from .plt_util import plt_temp, plt_vel

class VelTrainer:
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
            temp_loss = self.loss(pred[:, 0], label[:, 0])
            velx_loss = self.loss(pred[:, 1], label[:, 1])
            vely_loss = self.loss(pred[:, 2], label[:, 2])
            print(f'{temp_loss}, {velx_loss}, {vely_loss}')
            loss = (temp_loss + velx_loss + vely_loss) / 3
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            print(f'train loss: {loss}')
            del input, label

    def val_step(self, epoch):
        self.model.eval()
        for iter, (input, label) in enumerate(self.val_dataloader):
            input = input.cuda().float()
            label = label.cuda().float()
            with torch.no_grad():
                pred = self.model(input)
                temp_loss = F.mse_loss(pred[:, 0], label[:, 0])
                velx_loss = F.mse_loss(pred[:, 1], label[:, 1])
                vely_loss = F.mse_loss(pred[:, 2], label[:, 2])
                print(f'{temp_loss}, {velx_loss}, {vely_loss}')
                loss = (temp_loss + velx_loss + vely_loss) / 3
            print(f'val loss: {loss}')
            del input, label

    def test(self, dataset):
        self.model.eval()
        temp_preds = []
        velx_preds = []
        vely_preds = []
        temp_labels = []
        velx_labels = []
        vely_labels = []
        for timestep in range(len(dataset)):
            input, label = dataset[timestep]
            input = input.cuda().float().unsqueeze(0)
            label = label.cuda().float().unsqueeze(0)
            print(input.size(), label.size())
            with torch.no_grad():
                pred = self.model(input)
                temp = pred[:, 0]
                velx = F.hardtanh(pred[:, 1], min_val=-1, max_val=1)
                vely = F.hardtanh(pred[:, 2], min_val=-1, max_val=1)
                dataset.write_temp(temp, timestep)
                dataset.write_velx(velx, timestep)
                dataset.write_vely(vely, timestep)
                temp_preds.append(temp.detach().cpu())
                velx_preds.append(velx.detach().cpu())
                vely_preds.append(vely.detach().cpu())
                temp_labels.append(label[:, 0].detach().cpu())
                velx_labels.append(label[:, 1].detach().cpu())
                vely_labels.append(label[:, 2].detach().cpu())

        temp_preds = torch.cat(temp_preds, dim=0)
        velx_preds = torch.cat(velx_preds, dim=0)
        vely_preds = torch.cat(vely_preds, dim=0)
        temp_labels = torch.cat(temp_labels, dim=0)
        velx_labels = torch.cat(velx_labels, dim=0)
        vely_labels = torch.cat(vely_labels, dim=0)

        def mag(velx, vely):
            return torch.sqrt(velx**2 + vely**2)
        mag_preds = mag(velx_preds, vely_preds)
        mag_labels = mag(velx_labels, vely_labels)
        
        def print_metrics(pred, label):
            metrics = compute_metrics(pred, label, dataset.get_dfun().permute((2,0,1)))
            print(metrics)

        print('temp metrics:')
        print_metrics(temp_preds, temp_labels)
        print('velx metrics:')
        print_metrics(velx_preds, velx_labels)
        print('vely metrics:')
        print_metrics(vely_preds, vely_labels)
        print('mag metrics:')
        print_metrics(mag_preds, mag_labels)
        
        model_name = self.model.__class__.__name__
        plt_temp(temp_preds, temp_labels, model_name)
        max_mag = mag_labels.max()
        plt_vel(mag_preds, mag_labels, max_mag, model_name)
