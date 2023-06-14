from omegaconf import DictConfig, OmegaConf
import hydra
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
from neuralop.models import FNO, UNO
from pathlib import Path
import os
import time

from op_lib.hdf5_dataset import (
        TempInputDataset,
        VelDataset,
        TempVelDataset
)
from op_lib.unet import UNet2d 
from op_lib.temp_trainer import TempTrainer
from op_lib.vel_trainer import VelTrainer


torch_dataset_map = {
    'temp_input_dataset': TempInputDataset,
    'vel_dataset': TempVelDataset
}

model_map = {
    'unet2d': UNet2d,
    'fno': FNO
}

trainer_map = {
    'temp_input_dataset': TempTrainer,
    'vel_dataset': VelTrainer
}

def build_datasets(cfg):
    DatasetClass = torch_dataset_map[cfg.experiment.torch_dataset_name]
    time_window = cfg.experiment.train.time_window
    train_dataset = ConcatDataset([
        DatasetClass(p, transform=cfg.dataset.transform, time_window=time_window) for p in cfg.dataset.train_paths])
    val_dataset = ConcatDataset([
        DatasetClass(p, time_window=time_window) for p in cfg.dataset.val_paths])
    return train_dataset, val_dataset

def build_dataloaders(train_dataset, val_dataset, cfg):
    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=cfg.experiment.train.batch_size,
                                  shuffle=cfg.experiment.train.shuffle_data,
                                  num_workers=1,
                                  pin_memory=True)
    val_dataloader = DataLoader(val_dataset, 
                                batch_size=cfg.experiment.train.batch_size,
                                shuffle=False,
                                num_workers=1,
                                pin_memory=True)
    return train_dataloader, val_dataloader

def get_model(model_name, in_channels, out_channels):
    assert model_name in ('unet2d', 'fno', 'uno'), f'Model name {model_name} invalid'
    if model_name == 'unet2d': 
        model = UNet2d(in_channels=in_channels,
                       out_channels=out_channels,
                       init_features=64)
    elif model_name == 'fno':
        model = FNO(n_modes=(32, 32),
                    hidden_channels=64,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    n_layers=5)
    elif model_name == 'uno':
        model = UNO(in_channels=in_channels, 
                    out_channels=out_channels,
                    hidden_channels=64,
                    projection_channels=64,
                    uno_out_channels=[32,64,64,64,32],
                    uno_n_modes=[[32,32],[32,32],[32,32],[32,32],[32,32]],
                    uno_scalings=[[1.0,1.0],[1,1],[1,1],[1,1],[1,1]],
                    horizontal_skips_map=None,
                    n_layers=5)
    model = model.cuda().float()
    return model

@hydra.main(version_base=None, config_path='../conf', config_name='default')
def train_app(cfg):
    print(OmegaConf.to_yaml(cfg))
    print(cfg.dataset.train_paths)
    assert cfg.test or cfg.train

    job_id = os.getenv('SLURM_JOB_ID')
    if job_id:
        log_dir = f'{cfg.log_dir}/{job_id}'
    else:
        log_dir = f'{cfg.log_dir}'
    
    writer = SummaryWriter(log_dir=log_dir)

    train_dataset, val_dataset = build_datasets(cfg)
    train_dataloader, val_dataloader = build_dataloaders(train_dataset, val_dataset, cfg)
    print('train size: ', len(train_dataloader))
    tail = cfg.dataset.val_paths[0].split('-')[-1]
    print(tail, tail[:-5])
    val_variable = 0#int(tail[:-5])
    print('T_wall of val sim: ', val_variable)

    exp = cfg.experiment
    model_name = exp.model.model_name.lower()
    in_channels = train_dataset.datasets[0].in_channels
    out_channels = train_dataset.datasets[0].out_channels

    model = get_model(model_name, in_channels, out_channels)
    if cfg.model_checkpoint:
        model.load_state_dict(torch.load(cfg.model_checkpoint))
    print(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=exp.optimizer.initial_lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=exp.lr_scheduler.patience,
                                                   gamma=exp.lr_scheduler.factor)

    TrainerClass = trainer_map[exp.torch_dataset_name]
    trainer = TrainerClass(model,
                           train_dataloader,
                           val_dataloader,
                           optimizer,
                           lr_scheduler,
                           val_variable,
                           writer,
                           exp)
    print(trainer)

    if cfg.train and not cfg.model_checkpoint:
        trainer.train(exp.train.max_epochs)
        timestamp = int(time.time())
        ckpt_file = f'{model.__class__.__name__}_{exp.torch_dataset_name}_{exp.train.max_epochs}_{timestamp}.pt'
        ckpt_root = Path.home() / f'crsp/ai4ts/afeeney/thermal_models/{cfg.dataset.name}'
        Path(ckpt_root).mkdir(parents=True, exist_ok=True)
        ckpt_path = f'{ckpt_root}/{ckpt_file}'
        print(f'saving model to {ckpt_path}')
        torch.save(model.state_dict(), f'{ckpt_path}')

    if cfg.test:
        trainer.test(val_dataset.datasets[0])

if __name__ == '__main__':
    train_app()
