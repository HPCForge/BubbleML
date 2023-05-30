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

class VelTrainer:
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
            sizes = torch.tensor(range(64, input.size(-1) + 1, 64))
            size = sizes[torch.randperm(sizes.size(0))[0]].item()
            input = TF.resize(input, size)
            label = TF.resize(label, size)
            pred = self.model(input)
            #velx_loss = F.mse_loss(pred[:, 0], label[:, 0])
            #vely_loss = F.mse_loss(pred[:, 1], label[:, 1])
            velx_loss = self.loss(pred[:, 0], label[:, 0])
            vely_loss = self.loss(pred[:, 1], label[:, 1])
            loss = (velx_loss + vely_loss) / 2
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
                velx_loss = F.mse_loss(pred[:, 0], label[:, 0])
                vely_loss = F.mse_loss(pred[:, 1], label[:, 1])
                loss = (velx_loss + vely_loss) / 2
            print(f'val loss: {loss}')
            del input, label

    def test(self, dataset):
        self.model.eval()
        velx_preds = []
        vely_preds = []
        velx_labels = []
        vely_labels = []
        for timestep in range(len(dataset)):
            input, label = dataset[timestep]
            input = input.cuda().float().unsqueeze(0)
            label = label.cuda().float().unsqueeze(0)
            print(input.size(), label.size())
            with torch.no_grad():
                pred = self.model(input)
                velx = pred[:, 0]
                vely = pred[:, 1]
                dataset.write_velx(velx, timestep)
                dataset.write_vely(vely, timestep)
                velx_preds.append(velx.detach().cpu())
                vely_preds.append(vely.detach().cpu())
                velx_labels.append(label[:, 0].detach().cpu())
                vely_labels.append(label[:, 1].detach().cpu())

        velx_preds = torch.cat(velx_preds, dim=0)
        vely_preds = torch.cat(vely_preds, dim=0)
        velx_labels = torch.cat(velx_labels, dim=0)
        vely_labels = torch.cat(vely_labels, dim=0)

        def mag(velx, vely):
            return torch.sqrt(velx**2 + vely**2)
        mag_preds = mag(velx_preds, vely_preds)
        mag_labels = mag(velx_labels, vely_labels)
        
        def print_metrics(pred, label):
            metrics = compute_metrics(pred, label)
            print(metrics)

        print('velx metrics:')
        print_metrics(velx_preds, velx_labels)
        print('vely metrics:')
        print_metrics(vely_preds, vely_labels)
        print('mag metrics:')
        print_metrics(mag_preds, mag_labels)
        
        gt = mag_labels.max()
        for i in range(len(velx_preds)):
            i_str = str(i).zfill(3)

            def plt_temp_arr(f, ax, arr, mm, title):
                cm_object = ax.imshow(arr, vmin=0, vmax=mm, cmap='viridis')
                ax.title.set_text(title)
                ax.axis('off')
                return cm_object

            f, axarr = plt.subplots(2, 2, layout="constrained")
            pred_mag = mag_preds[i]
            label_mag = mag_labels[i] 

            cm_object = plt_temp_arr(f, axarr[0, 0], np.flipud(label_mag), gt, 'Ground Truth')
            plt_temp_arr(f, axarr[0, 1], np.flipud(pred_mag), gt, 'ML Model')
            f.colorbar(cm_object, ax=axarr[0, 1], fraction=0.05)

            cm_object = plt_temp_arr(f, axarr[1, 0], np.flipud(np.abs(label_mag - pred_mag)), 1, 'Absolute Error')
            f.colorbar(cm_object, ax=axarr[1, 0], fraction=0.05)

            vx_pred, vx_label = velx_preds[i], velx_labels[i]
            vy_pred, vy_label = vely_preds[i], vely_labels[i]
            xd = (vx_pred - vx_label) ** 2
            yd = (vy_pred - vy_label) ** 2
            n = np.sqrt(xd + yd)
            cm_object = plt_temp_arr(f, axarr[1, 1], np.flipud(n), 1, 'L2 Error')
            f.colorbar(cm_object, ax=axarr[1, 1], fraction=0.05)

            plt.savefig(f'test_im/vel/{i_str}.png', dpi=500)
            plt.close()
