from omegaconf import DictConfig, OmegaConf
import hydra
import torch
from torch import nn
import torchvision
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Sampler
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
import time
import math

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from op_lib.disk_hdf5_dataset import (
        DiskTempInputDataset,
        DiskTempVelDataset
)
from op_lib.hdf5_dataset import (
        HDF5ConcatDataset,
        TempInputDataset,
        TempVelDataset
)

from op_lib.temp_trainer import TempTrainer
from op_lib.vel_trainer import VelTrainer
from op_lib.push_vel_trainer import PushVelTrainer
from op_lib import dist_utils

from models.get_model import get_model


torch_dataset_map = {
    'temp_input_dataset': (DiskTempInputDataset, TempInputDataset),
    'vel_dataset': (DiskTempVelDataset, TempVelDataset)
}

trainer_map = {
    'temp_input_dataset': TempTrainer,
    'vel_dataset': PushVelTrainer
}

class LinearWarmupLR(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup_iters):
        self.warmup_iters = warmup_iters
        warmup_func = lambda current_step: min(1, current_step / self.warmup_iters)
        super().__init__(optimizer, lr_lambda=warmup_func)

def build_datasets(cfg):
    DatasetClass = torch_dataset_map[cfg.experiment.torch_dataset_name]
    time_window = cfg.experiment.train.time_window
    future_window = cfg.experiment.train.future_window
    push_forward_steps = cfg.experiment.train.push_forward_steps 
    use_coords = cfg.experiment.train.use_coords
    steady_time = cfg.dataset.steady_time

    # normalize temperatures and velocities to [-1, 1]
    train_dataset = HDF5ConcatDataset([
        DatasetClass[0](p,
                        steady_time=cfg.dataset.steady_time,
                        use_coords=use_coords,
                        transform=cfg.dataset.transform,
                        time_window=time_window,
                        future_window=future_window,
                        push_forward_steps=push_forward_steps) for p in cfg.dataset.train_paths])
    train_max_temp = train_dataset.normalize_temp_()
    train_max_vel = train_dataset.normalize_vel_()

    # use same mapping as train dataset to normalize validation set
    val_dataset = HDF5ConcatDataset([
        DatasetClass[1](p,
                        steady_time=cfg.dataset.steady_time,
                        use_coords=use_coords,
                        time_window=time_window,
                        future_window=future_window) for p in cfg.dataset.val_paths])
    val_dataset.normalize_temp_(train_max_temp)
    val_dataset.normalize_vel_(train_max_vel)

    assert val_dataset.absmax_temp() <= 1.5
    assert val_dataset.absmax_vel() <= 1.5
    return train_dataset, val_dataset

def build_dataloaders(train_dataset, val_dataset, cfg):
    if cfg.experiment.distributed:
        train_sampler = DistributedSampler(dataset=train_dataset,
                                           shuffle=cfg.experiment.train.shuffle_data)
        val_sampler = DistributedSampler(dataset=val_dataset,
                                         shuffle=False)
    else:
        train_sampler, val_sampler = None, None
    
    train_shuffle = cfg.experiment.train.shuffle_data and (train_sampler is None)
    train_dataloader = DataLoader(train_dataset, 
                                  sampler=train_sampler,
                                  shuffle=train_shuffle,
                                  batch_size=cfg.experiment.train.batch_size,
                                  num_workers=2,
                                  pin_memory=True,
                                  prefetch_factor=4)
    val_dataloader = DataLoader(val_dataset, 
                                sampler=val_sampler,
                                batch_size=cfg.experiment.train.batch_size,
                                shuffle=False,
                                num_workers=2,
                                pin_memory=True,
                                prefetch_factor=4)
    return train_dataloader, val_dataloader

def nparams(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

@hydra.main(version_base=None, config_path='../conf', config_name='default')
def train_app(cfg):
    print(OmegaConf.to_yaml(cfg))
    print(cfg.dataset.train_paths)
    assert cfg.test or cfg.train
    assert cfg.data_base_dir is not None
    assert cfg.log_dir is not None
    assert cfg.experiment.train.time_window > 0
    assert cfg.experiment.train.future_window > 0
    assert cfg.experiment.train.push_forward_steps > 0

    if cfg.experiment.distributed:
        dist_utils.initialize('nccl')

    job_id = os.getenv('SLURM_JOB_ID')
    if job_id:
        log_dir = f'{cfg.log_dir}/{job_id}'
    else:
        log_dir = f'{cfg.log_dir}'
    
    writer = SummaryWriter(log_dir=log_dir)

    train_dataset, val_dataset = build_datasets(cfg)
    train_dataloader, val_dataloader = build_dataloaders(train_dataset, val_dataset, cfg)
    print('train size: ', len(train_dataloader))
    #tail = cfg.dataset.val_paths[0].split('-')[-1]
    #print(tail, tail[:-5])
    #val_variable = int(tail[:-5])
    #print('T_wall of val sim: ', val_variable)
    val_variable = 0

    exp = cfg.experiment
    model_name = exp.model.model_name.lower()
    in_channels = train_dataset.datasets[0].in_channels
    out_channels = train_dataset.datasets[0].out_channels

    model = get_model(model_name, in_channels, out_channels, exp)

    if cfg.model_checkpoint:
        model.load_state_dict(torch.load(cfg.model_checkpoint))
    print(model)
    np = nparams(model)
    print(f'Model has {np} parameters')

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=dist_utils.world_size() * exp.optimizer.initial_lr,
                                  weight_decay=exp.optimizer.weight_decay)

    total_iters = exp.train.max_epochs * len(train_dataloader)
    warmup_iters = max(1, int(math.sqrt(dist_utils.world_size()) * 0.03 * total_iters))
    warmup_lr = LinearWarmupLR(optimizer, warmup_iters)
    warm_iters = total_iters - warmup_iters
    if exp.lr_scheduler.name == 'step':
        warm_schedule = torch.optim.lr_scheduler.StepLR(optimizer,
                                                        step_size=exp.lr_scheduler.patience * len(train_dataloader),
                                                        gamma=exp.lr_scheduler.factor)
    elif exp.lr_scheduler.name == 'cosine':
        warm_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                   T_max=exp.train.max_epochs * len(train_dataloader),
                                                                   eta_min=1e-6)
    # SequentialLR produces a deprecation warning when calling sub-schedulers.
    # https://github.com/pytorch/pytorch/issues/76113
    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup_lr, warm_schedule], [warmup_iters])

    TrainerClass = trainer_map[exp.torch_dataset_name]
    trainer = TrainerClass(model,
                           exp.train.future_window,
                           exp.train.push_forward_steps,
                           train_dataloader,
                           val_dataloader,
                           optimizer,
                           lr_scheduler,
                           val_variable,
                           writer,
                           exp)
    print(trainer)

    if cfg.train and not cfg.model_checkpoint:
        trainer.train(exp.train.max_epochs, log_dir, dataset_name=cfg.dataset.name)
        timestamp = int(time.time())
        if cfg.experiment.distributed:
            model_name = model.module.__class__.__name__
        else:
            model_name = model.__class__.__name__
        ckpt_file = f'{model_name}_{exp.torch_dataset_name}_{exp.train.max_epochs}_{timestamp}.pt'
        ckpt_root = Path.home() / f'{log_dir}/{cfg.dataset.name}'
        Path(ckpt_root).mkdir(parents=True, exist_ok=True)
        ckpt_path = f'{ckpt_root}/{ckpt_file}'
        print(f'saving model to {ckpt_path}')
        if cfg.experiment.distributed:
            torch.save(model.module.state_dict(), f'{ckpt_path}')
        else:
            torch.save(model.state_dict(), f'{ckpt_path}')

    if cfg.test and dist_utils.is_leader_process():
        trainer.test(val_dataset.datasets[0])

if __name__ == '__main__':
    train_app()
