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
from pathlib import Path

from .hdf5_dataset import HDF5Dataset, VelCoordInputDataset
from .metrics import compute_metrics, write_metrics
from .losses import LpLoss
from .plt_util import plt_temp, plt_iter_mae, plt_vel
from .heatflux import heatflux
from .dist_utils import local_rank, is_leader_process
from .downsample import downsample_domain

from torch.cuda import nvtx 

t_bulk_map = {
    'wall_super_heat': 58,
    'subcooled': 50
}

class VelCoordTrainer:
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

    def push_forward_prob(self, epoch, max_epochs):
        r"""
        Randomly set the number of push-forward steps based on current
        iteration. Initially, it's unlike to "push-forward." later in training,
        it's nearly certain to apply the push-forward trick.
        """
        cur_iter = epoch * len(self.train_dataloader)
        tot_iter = max_epochs * len(self.train_dataloader)
        frac = cur_iter / tot_iter
        if np.random.uniform() > frac:
            return 1
        else:
            return self.max_push_forward_steps

    def train(self, max_epochs, log_dir, dataset_name):
        for epoch in range(max_epochs):
            print('epoch ', epoch)
            self.train_step(epoch, max_epochs)
            self.val_step(epoch)
            if is_leader_process():
                val_dataset = self.val_dataloader.dataset.datasets[0]
                self.test(val_dataset)
                self.save_checkpoint(log_dir, dataset_name)

    def _forward_int(self, coords, vel, dfun):
        # TODO: account for possibly different timestep sizes of training data
        # Print the shapes of the tensors
    
        # Optional: Print the actual data (if the tensors are not too large)
        input = torch.cat((vel, dfun), dim=1)
        if self.use_coords:
            input = torch.cat((coords, input), dim=1)
        pred = self.model(input)

        #timesteps = (torch.arange(self.future_window) + 1).cuda().unsqueeze(-1).unsqueeze(-1).float()
        #timesteps /= 10 # timestep size is 0.1 for vel
        #timesteps = timesteps.to(pred.device)

        #d_temp = pred[:, :self.future_window]
        #last_temp_input = temp[:, -1].unsqueeze(1)
        #temp_pred = last_temp_input + timesteps * d_temp

        #d_vel = pred[:, self.future_window:]
        #last_vel_input = vel[:, -2:].repeat(1, self.future_window, 1, 1)
        #timesteps_interleave = torch.repeat_interleave(timesteps, 2, dim=0) 
        #vel_pred = last_vel_input + timesteps_interleave * d_vel

        

        return pred

    def _index_push(self, idx,coords, vel, dfun):
        r"""
        select the channels for push_forward_step `idx`
        
        """
       # print("Shape1 of vel:", vel.shape)
       # print("Shape1 of dfun:", dfun.shape)
       # print("Shape2 of vel:", vel[:, idx].shape)
       # print("Shape2 of dfun:", dfun[:, idx].shape)
        return (coords[:, idx],vel[:, idx], dfun[:, idx])
        #return (vel, dfun)

    def _index_dfun(self, idx, dfun):
        return dfun[:, idx]

    def push_forward_trick(self,coords, vel, dfun, push_forward_steps):
        # TODO: clean this up...
        coords_input, vel_input, dfun_input = self._index_push(0,coords,vel, dfun)
        #print("Expected future window size:", self.future_window)
        #print("Actual size of vel_input:", vel_input.size(1))
        
        assert self.future_window*2 == vel_input.size(1), 'push-forward expects history size to match future'
        coords_input, vel_input, dfun_input = \
                downsample_domain(self.cfg.train.downsample_factor,coords_input, vel_input, dfun_input)
        with torch.no_grad():
            for idx in range(push_forward_steps - 1):
                vel_input = self._forward_int(coords_input, vel_input, dfun_input)
                dfun_input = self._index_dfun(idx + 1, dfun)
                dfun_input = downsample_domain(self.cfg.train.downsample_factor, dfun_input)[0]
        if self.cfg.train.noise and push_forward_steps == 1:
            vel_input += torch.empty_like(vel_input).normal_(0, 0.01)
        vel_pred = self._forward_int(coords_input, vel_input, dfun_input)
        return  vel_pred

    def train_step(self, epoch, max_epochs):
        self.model.train()

        # warmup before doing push forward trick
        for iter, (coords, vel, dfun,vel_label) in enumerate(self.train_dataloader):
            coords = coords.to(local_rank()).float()
            vel = vel.to(local_rank()).float()
            dfun = dfun.to(local_rank()).float()
            push_forward_steps = self.push_forward_prob(epoch, max_epochs)
            
            vel_pred = self.push_forward_trick(coords, vel, dfun, push_forward_steps)

            idx = (push_forward_steps - 1)
            vel_label = vel_label[:, idx].to(local_rank()).float()
          #  print("Shape1 of vel_label:", vel_label.shape)
            vel_label = downsample_domain(self.cfg.train.downsample_factor, vel_label)[0]
           

            vel_loss = F.mse_loss(vel_pred, vel_label)
            loss = vel_loss
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
        for iter, (coords, vel, dfun, vel_label) in enumerate(self.val_dataloader):
            coords = coords.to(local_rank()).float()
            vel = vel.to(local_rank()).float()
            dfun = dfun.to(local_rank()).float()

            # val doesn't apply push-forward
            vel_label = vel_label[:, 0].to(local_rank()).float()

            with torch.no_grad():
                vel_pred = self._forward_int(coords[:, 0],vel[:, 0], dfun[:, 0])
                vel_loss = F.mse_loss(vel_pred, vel_label)
                loss = vel_loss
            print(f'val loss: {loss}')
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
                vel_pred = self._forward_int(coords[:, 0],vel[:, 0], dfun[:, 0])
                vel_pred = vel_pred.squeeze(0)
                dataset.write_vel(vel_pred, timestep)
                vels.append(vel_pred.detach().cpu())
                vels_labels.append(vel_label.detach().cpu())

        vels = torch.cat(vels, dim=0)
        vels_labels = torch.cat(vels_labels, dim=0)
        dfun = dataset.get_dfun()[:vels.size(0)//2]

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