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

class TempTrainer:
    def __init__(self,
                 model,
                 train_dataloader,
                 val_dataloader,
                 optimizer,
                 lr_scheduler,
                 writer,
                 cfg):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.writer = writer
        self.cfg = cfg

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
            sizes = torch.tensor(range(64, 513, 64))
            size = sizes[torch.randperm(sizes.size(0))[0]].item()
            input = TF.resize(input, size)
            label = TF.resize(label, size)

            pred = self.model(input)
            temp_loss = F.mse_loss(pred, label)
            loss = temp_loss
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
                temp_loss = F.mse_loss(pred, label)
                loss = temp_loss
            print(f'val loss: {loss}')
            plt.imsave(f'im/val_label_{iter}.png', np.flipud(label[0, 0].detach().cpu()))
            plt.imsave(f'im/val_pred_{iter}.png', np.flipud(pred[0, 0].detach().cpu()))
            del input, label
