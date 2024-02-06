import torch
from torch import nn
import torchvision
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, PolynomialLR
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms.functional as TF
from neuralop.layers.resample import resample

import matplotlib.pyplot as plt
import numpy as np
import time
from pathlib import Path

from .hdf5_dataset import HDF5Dataset, TempVelDataset, VelPDEInputDataset
from .metrics import compute_metrics, write_metrics
from .losses import LpLoss
from .plt_util import plt_temp, plt_vel
from .pde_losses import Vel_PDE_Loss
from .dist_utils import local_rank, is_leader_process
from .downsample import downsample_domain
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

    def save_checkpoint(self, dataset_name):
        timestamp = int(time.time())
        if self.cfg.distributed:
            model_name = self.model.module.__class__.__name__
        else:
            model_name = self.model.__class__.__name__
        ckpt_file = f'{model_name}_{cfg.torch_dataset_name}_{cfg.train.max_epochs}_{timestamp}.pt'
        ckpt_root = Path.home() / f'{log_dir}/{dataset_name}'
        Path(ckpt_root).mkdir(parents=True, exist_ok=True)
        ckpt_path = f'{ckpt_root}/{ckpt_file}'
        print(f'saving model to {ckpt_path}')
        if cfg.distributed:
            torch.save(self.model.module.state_dict(), f'{ckpt_path}')
        else:
            torch.save(self.model.state_dict(), f'{ckpt_path}')

    def train(self, max_epochs, dataset_name):
        for epoch in range(max_epochs):
            print('epoch ', epoch)
            self.train_step(epoch)
            self.val_step(epoch)
            self.lr_scheduler.step()
            val_dataset = self.val_dataloader.dataset.datasets[0]
            self.test(val_dataset)
            self.save_checkpoint(dataset_name)

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

        return metrics

class VelPDETrainer:
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

    def _forward_int(self, coords, vel, dfun):
        input = torch.cat((vel, dfun), dim=1)
        if self.cfg.train.use_coords:
            input = torch.cat((coords, input), dim=1)
        pred = self.model(input)
        return pred

    def save_checkpoint(self, log_dir, dataset_name):
        timestamp = int(time.time())
        if self.cfg.distributed:
            model_name = self.model.module.__class__.__name__
        else:
            model_name = self.model.__class__.__name__
        ckpt_file = f'{model_name}_{self.cfg.torch_dataset_name}_{self.cfg.train.max_epochs}_{timestamp}.pt'
        ckpt_root = Path.home() / f'{log_dir}/{dataset_name}'
        Path(ckpt_root).mkdir(parents=True, exist_ok=True)
        ckpt_path = f'{ckpt_root}/{ckpt_file}'
        print(f'saving model to {ckpt_path}')
        if self.cfg.distributed:
            torch.save(self.model.module.state_dict(), f'{ckpt_path}')
        else:
            torch.save(self.model.state_dict(), f'{ckpt_path}')

    def train(self, max_epochs, log_dir, dataset_name):
        for epoch in range(max_epochs):
            print('epoch ', epoch)
            self.train_step(epoch, max_epochs)
            self.val_step(epoch)
            if is_leader_process():
                val_dataset = self.val_dataloader.dataset.datasets[0]
                self.test(val_dataset)
                self.save_checkpoint(log_dir, dataset_name)

    def train_step(self, epoch, max_epochs):
        self.model.train()

        # warmup before doing push forward trick
        for iter, (coords, vel, dfun, vel_label, unnormalized_pressure, dfun_future, run_time_params, resolution) in enumerate(self.train_dataloader):
            PDE_loss_class = Vel_PDE_Loss(run_time_params, self.cfg.optimizer.temporal_decay_factor)

            resolution = [r[0] for r in resolution]
            
            coords = coords.to(local_rank()).float()
            vel = vel.to(local_rank()).float()
            dfun = dfun.to(local_rank()).float()
            vel_label = vel_label.to(local_rank()).float()
            unnormalized_pressure = unnormalized_pressure.to(local_rank()).float()
            dfun_future = dfun_future.to(local_rank()).float()


            vel_pred = self._forward_int(coords, vel, dfun)
            
            vel_x = vel_pred[:, 1]
            vel_y = vel_pred[:, 2]

            velx_loss = self.loss(vel_x, vel_label[:, 1])
            vely_loss = self.loss(vel_y, vel_label[:, 2])
            vel_pde_loss = PDE_loss_class(unnormalized_pressure, vel_x, vel_y, dfun_future, resolution)

            print(vel_pred.size(), vel_label.size())

            loss = (velx_loss + vely_loss + vel_pde_loss) / 3
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
            
            print(f'train loss: {loss}')
            global_iter = epoch * len(self.train_dataloader) + iter
            write_metrics(vel_pred, vel_label, global_iter, 'TrainVel', self.writer)
            del vel, vel_label

    def val_step(self, epoch):
        self.model.eval()
        for iter, (coords, vel, dfun, vel_label, unnormalized_pressure, dfun_future, run_time_params, resolution) in enumerate(self.val_dataloader):
            PDE_loss_class = Vel_PDE_Loss(run_time_params, self.cfg.optimizer.temporal_decay_factor)
            resolution = [r[0] for r in resolution]

            coords = coords.to(local_rank()).float()
            vel = vel.to(local_rank()).float()
            dfun = dfun.to(local_rank()).float()
            vel_label = vel_label.to(local_rank()).float()
            unnormalized_pressure = unnormalized_pressure.to(local_rank()).float()
            dfun_future = dfun_future.to(local_rank()).float()

            print(vel_pred.shape)
            vel_x = vel_pred[:, 0]
            vel_y = vel_pred[:, 1]

            with torch.no_grad():
                vel_pred = self._forward_int(coords, vel)
                velx_loss = self.loss(vel_x, vel_label[:, 0])
                vely_loss = self.loss(vel_y, vel_label[:, 1])
                vel_pde_loss = PDE_loss_class(unnormalized_pressure, vel_x, vel_y, dfun_future, resolution)
           
            print(f'val loss: {velx_loss+vely_loss}, val pde loss: {vel_pde_loss}')
            global_iter = epoch * len(self.val_dataloader) + iter
            write_metrics(vel_pred, vel_label, global_iter, 'ValVel', self.writer)
            del vel, vel_label

    def test(self, dataset, max_time_limit=200):
        self.model.eval()
        vels = []
        vels_labels = []
        time_limit = min(max_time_limit, len(dataset))
        for timestep in range(0, time_limit, self.future_window):
            coords, vel, dfun, vel_label = dataset[timestep]
            coords = coords.to(local_rank()).float().unsqueeze(0)
            vel = vel.to(local_rank()).float().unsqueeze(0)
            dfun = dfun.to(local_rank()).float().unsqueeze(0)
            # val doesn't apply push-forward
            vel_label = vel_label[0].to(local_rank()).float()
            with torch.no_grad():
                vel_pred = self._forward_int(coords[:, 0], vel[:, 0], dfun[:, 0])
                vel_pred = vel_pred.squeeze(0)
                dataset.write_vel(vel_pred, timestep)
                vels.append(vel_pred.detach().cpu())
                vels_labels.append(vel_label.detach().cpu())

        vels = torch.cat(vels, dim=0)
        vels_labels = torch.cat(vels_labels, dim=0)
        dfun = dataset.get_dfun()[:vels.size(0)]

        print(vels.size(), vels_labels.size(), dfun.size())

        velx_preds = vels[0::2]
        velx_labels = vels_labels[0::2]
        vely_preds = vels[1::2]
        vely_labels = vels_labels[1::2]

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
        
        def mag(velx, vely):
            return torch.sqrt(velx**2 + vely**2)
        mag_preds = mag(velx_preds, vely_preds)
        mag_labels = mag(velx_labels, vely_labels)

        plt_vel(mag_preds, mag_labels,
                velx_preds, velx_labels,
                vely_preds, vely_labels,
                model_name)

        dataset.reset()