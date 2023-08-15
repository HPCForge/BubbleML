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
from .plt_util import plt_temp, plt_iter_mae, plt_vel
from .heatflux import heatflux
from .dist_utils import local_rank

from torch.cuda import nvtx 

t_bulk_map = {
    'wall_super_heat': 58,
    'subcooled': 50
}

class PushVelTrainer:
    def __init__(self,
                 model,
                 future_window,
                 max_push_forward_steps,
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
        self.loss = LpLoss(d=2, reduce_dims=[0, 1])

        self.max_push_forward_steps = max_push_forward_steps
        self.future_window = future_window
        self.use_coords = cfg.train.use_coords

    def train(self, max_epochs):
        for epoch in range(max_epochs):
            print('epoch ', epoch)
            self.train_step(epoch)
            self.val_step(epoch)

    def _forward_int(self, coords, temp, vel, dfun):
        # TODO: account for possibly different timestep sizes of training data
        input = torch.cat((temp, vel, dfun), dim=1)
        if self.use_coords:
            input = torch.cat((coords, input), dim=1)
        pred = self.model(input)

        timesteps = (torch.arange(self.future_window) + 1).cuda().unsqueeze(-1).unsqueeze(-1).float()
        timesteps /= 10 # timestep size is 0.1 for vel
        timesteps = timesteps.to(pred.device)

        d_temp = pred[:, :self.future_window]
        last_temp_input = temp[:, -1].unsqueeze(1)
        temp_pred = last_temp_input + timesteps * d_temp

        d_vel = pred[:, self.future_window:]
        last_vel_input = vel[:, -2:].repeat(1, self.future_window, 1, 1)
        timesteps_interleave = torch.repeat_interleave(timesteps, 2, dim=0) 
        vel_pred = last_vel_input + timesteps_interleave * d_vel

        #temp_pred = pred[:, :self.future_window]
        #vel_pred = pred[:, self.future_window:]

        return temp_pred, vel_pred

    def _index_push(self, idx, coords, temp, vel, dfun):
        r"""
        select the channels for push_forward_step `idx`
        """
        return (coords[:, idx], temp[:, idx], vel[:, idx], dfun[:, idx])

    def _index_dfun(self, idx, dfun):
        return dfun[:, idx]

    def push_forward_trick(self, coords, temp, vel, dfun, push_forward_steps):
        # TODO: clean this up...
        coords_input, temp_input, vel_input, dfun_input = self._index_push(0, coords, temp, vel, dfun)
        assert self.future_window == temp_input.size(1), 'push-forward expects history size to match future'
        coords_input, temp_input, vel_input, dfun_input = \
                self.downsample_domain(coords_input, temp_input, vel_input, dfun_input)
        with torch.no_grad():
            for idx in range(push_forward_steps - 1):
                temp_input, vel_input = self._forward_int(coords_input, temp_input, vel_input, dfun_input)
                dfun_input = self._index_dfun(idx + 1, dfun)
                dfun_input = self.downsample_domain(dfun)[0]
        if self.cfg.train.noise and push_forward_steps == 1:
            temp_input += torch.empty_like(temp_input).normal_(0, 0.01)
            vel_input += torch.empty_like(vel_input).normal_(0, 0.01)
        temp_pred, vel_pred = self._forward_int(coords_input, temp_input, vel_input, dfun_input)
        return temp_pred, vel_pred

    def downsample_domain(self, *args):
        downsample_factor = self.cfg.train.downsample_factor
        assert downsample_factor > 0 and downsample_factor <= 1.0
        return tuple([F.interpolate(im, scale_factor=downsample_factor) for im in args])

    def train_step(self, epoch):
        self.model.train()

        # warmup before doing push forward trick
        push_forward_steps = 1 if epoch < 3 else self.max_push_forward_steps

        for iter, (coords, temp, vel, dfun, temp_label, vel_label) in enumerate(self.train_dataloader):
            coords = coords.to(local_rank()).float()
            temp = temp.to(local_rank()).float()
            vel = vel.to(local_rank()).float()
            dfun = dfun.to(local_rank()).float()
            
            temp_pred, vel_pred = self.push_forward_trick(coords, temp, vel, dfun, push_forward_steps)

            idx = (push_forward_steps - 1)
            temp_label = temp_label[:, idx].to(local_rank()).float()
            idx = (push_forward_steps - 1)
            vel_label = vel_label[:, idx].to(local_rank()).float()

            temp_label, vel_label = self.downsample_domain(temp_label, vel_label)

            temp_loss = self.loss(temp_pred, temp_label)
            vel_loss = self.loss(vel_pred, vel_label)
            loss = (temp_loss + vel_loss) / 2
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
            
            print(f'train loss: {loss}')
            global_iter = epoch * len(self.train_dataloader) + iter
            write_metrics(temp_pred, temp_label, global_iter, 'TrainTemp', self.writer)
            write_metrics(vel_pred, vel_label, global_iter, 'TrainVel', self.writer)
            del temp, vel, temp_label, vel_label

    def val_step(self, epoch):
        self.model.eval()
        for iter, (coords, temp, vel, dfun, temp_label, vel_label) in enumerate(self.val_dataloader):
            coords = coords.to(local_rank()).float()
            temp = temp.to(local_rank()).float()
            vel = vel.to(local_rank()).float()
            dfun = dfun.to(local_rank()).float()

            # val doesn't apply push-forward
            temp_label = temp_label[:, 0].to(local_rank()).float()
            vel_label = vel_label[:, 0].to(local_rank()).float()

            with torch.no_grad():
                temp_pred, vel_pred = self._forward_int(coords[:, 0], temp[:, 0], vel[:, 0], dfun[:, 0])
                temp_loss = F.mse_loss(temp_pred, temp_label)
                vel_loss = F.mse_loss(vel_pred, vel_label)
                loss = (temp_loss + vel_loss) / 2
            print(f'val loss: {loss}')
            global_iter = epoch * len(self.val_dataloader) + iter
            write_metrics(temp_pred, temp_label, global_iter, 'ValTemp', self.writer)
            write_metrics(vel_pred, vel_label, global_iter, 'ValVel', self.writer)
            del temp, vel, temp_label, vel_label

    def test(self, dataset, max_time_limit=200):
        self.model.eval()
        temps = []
        temps_labels = []
        vels = []
        vels_labels = []
        time_limit = min(max_time_limit, len(dataset))
        for timestep in range(0, time_limit, self.future_window):
            coords, temp, vel, dfun, temp_label, vel_label = dataset[timestep]
            coords = coords.to(local_rank()).float().unsqueeze(0)
            temp = temp.to(local_rank()).float().unsqueeze(0)
            vel = vel.to(local_rank()).float().unsqueeze(0)
            dfun = dfun.to(local_rank()).float().unsqueeze(0)
            # val doesn't apply push-forward
            temp_label = temp_label[0].to(local_rank()).float()
            vel_label = vel_label[0].to(local_rank()).float()
            with torch.no_grad():
                temp_pred, vel_pred = self._forward_int(coords[:, 0], temp[:, 0], vel[:, 0], dfun[:, 0])
                temp_pred = temp_pred.squeeze(0)
                vel_pred = vel_pred.squeeze(0)
                dataset.write_temp(temp_pred, timestep)
                dataset.write_vel(vel_pred, timestep)
                temps.append(temp_pred.detach().cpu())
                temps_labels.append(temp_label.detach().cpu())
                vels.append(vel_pred.detach().cpu())
                vels_labels.append(vel_label.detach().cpu())

        temps = torch.cat(temps, dim=0)
        temps_labels = torch.cat(temps_labels, dim=0)
        vels = torch.cat(vels, dim=0)
        vels_labels = torch.cat(vels_labels, dim=0)
        dfun = dataset.get_dfun()[:temps.size(0)]

        print(temps.size(), temps_labels.size(), dfun.size())
        print(vels.size(), vels_labels.size(), dfun.size())

        velx_preds = vels[0::2]
        velx_labels = vels_labels[0::2]
        vely_preds = vels[1::2]
        vely_labels = vels_labels[1::2]

        print(temps.size(), temps_labels.size(), dfun.size())

        metrics = compute_metrics(temps, temps_labels, dfun)
        print('TEMP METRICS')
        print(metrics)
        metrics = compute_metrics(velx_preds, velx_labels, dfun)
        print('VELX METRICS')
        print(metrics)
        metrics = compute_metrics(vely_preds, vely_labels, dfun)
        print('VELY METRICS')
        print(metrics)
        
        #xgrid = dataset.get_x().permute((2, 0, 1))
        #print(heatflux(temps, dfun, self.val_variable, xgrid, dataset.get_dy()))
        #print(heatflux(labels, dfun, self.val_variable, xgrid, dataset.get_dy()))
        
        model_name = self.model.__class__.__name__
        plt_iter_mae(temps, temps_labels)
        plt_temp(temps, temps_labels, model_name)

        def mag(velx, vely):
            return torch.sqrt(velx**2 + vely**2)
        mag_preds = mag(velx_preds, vely_preds)
        mag_labels = mag(velx_labels, vely_labels)

        max_mag = mag_labels.max()
        plt_vel(mag_preds, mag_labels, max_mag, model_name)
