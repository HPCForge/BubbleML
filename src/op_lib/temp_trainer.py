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
            #sizes = torch.tensor(range(128, input.size(-1) + 1, 64))
            #size = sizes[torch.randperm(sizes.size(0))[0]].item()
            #input = TF.resize(input, size)
            #label = TF.resize(label, size)

            pred = self.model(input)
            #temp_loss = F.mse_loss(pred, label)
            temp_loss = self.loss(pred, label)
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
            del input, label

    def test(self, dataset):
        self.model.eval()
        temps = []
        labels = []
        for timestep in range(len(dataset)):
            input, label = dataset[timestep]
            input = input.cuda().float().unsqueeze(0)
            label = label.cuda().float().unsqueeze(0)
            with torch.no_grad():
                pred = self.model(input)
                temp = F.hardtanh(pred[:, 0], min_val=0, max_val=1)
                dataset.write_temp(temp, timestep)
                temps.append(temp.detach().cpu())
                labels.append(label[:, 0].detach().cpu())
        temps = torch.cat(temps, dim=0)
        labels = torch.cat(labels, dim=0)
        metrics = compute_metrics(temps, labels, dataset.get_dfun().permute((2, 0, 1)))
        print(metrics)
        
        for i in range(len(temps)):
            i_str = str(i).zfill(3)

            def plt_temp_arr(f, ax, arr, mm, title):
                cm_object = ax.imshow(arr, vmin=0, vmax=mm, cmap='plasma')
                ax.title.set_text(title)
                ax.axis('off')
                return cm_object

            temp = temps[i].numpy()
            label = labels[i].numpy()
            f, axarr = plt.subplots(1, 3, layout="constrained")
            cm_object = plt_temp_arr(f, axarr[0], np.flipud(label), 1, 'Ground Truth')
            plt_temp_arr(f, axarr[1], np.flipud(temp), 1, 'ML Model')
            f.colorbar(cm_object, ax=axarr[1], fraction=0.05)
            
            err = np.abs(temp - label)
            cm_object = plt_temp_arr(f, axarr[2], np.flipud(err), 1, 'Absolute Error')
            f.colorbar(cm_object, ax=axarr[2], fraction=0.05)

            f.set_size_inches(w=8, h=3)
            plt.savefig(f'test_im/temp/{i_str}.png', dpi=600, transparent=True)
            plt.close()
