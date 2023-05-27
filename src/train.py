from omegaconf import DictConfig, OmegaConf
import hydra
from hdf5_dataset import HDF5Dataset, TempDataset, TempInputDataset, VelDataset
import torch
from torch import nn
import torchvision
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, PolynomialLR
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import apex
from unet import UNet2d 
import numpy as np
from neuralop.models import FNO
from temp_trainer import TempTrainer
from vel_trainer import VelTrainer
from pathlib import Path
import os

torch_dataset_map = {
    'temp_dataset': TempDataset,
    'temp_input_dataset': TempInputDataset,
    'vel_dataset': VelDataset
}

model_map = {
    'unet2d': UNet2d,
    'fno': FNO
}

trainer_map = {
    'temp_dataset': TempTrainer,
    'temp_input_dataset': TempTrainer,
    'vel_dataset': VelTrainer
}

def build_datasets(cfg):
    DatasetClass = torch_dataset_map[cfg.torch_dataset_name]
    train_dataset = ConcatDataset([
        DatasetClass(p, transform=True, time_window=cfg.train.time_window) for p in cfg.dataset.train_paths])
    val_dataset = ConcatDataset([
        DatasetClass(p, time_window=cfg.train.time_window) for p in cfg.dataset.val_paths])
    return train_dataset, val_dataset

def build_dataloaders(train_dataset, val_dataset, cfg):
    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=cfg.train.batch_size,
                                  shuffle=cfg.train.shuffle_data,
                                  num_workers=1,
                                  pin_memory=True)
    val_dataloader = DataLoader(val_dataset, 
                                batch_size=cfg.train.batch_size,
                                shuffle=False,
                                num_workers=1,
                                pin_memory=True)
    return train_dataloader, val_dataloader

@hydra.main(version_base=None, config_path='../conf', config_name='default')
def train_app(cfg):
    print(OmegaConf.to_yaml(cfg))

    writer = SummaryWriter(log_dir=cfg.log.log_dir)

    train_dataset, val_dataset = build_datasets(cfg)
    train_dataloader, val_dataloader = build_dataloaders(train_dataset, val_dataset, cfg)
    print('train size: ', len(train_dataloader))

    model_name = cfg.model.model_name.lower()
    in_channels = train_dataset.datasets[0].in_channels
    out_channels = train_dataset.datasets[0].out_channels

    assert model_name in ('unet2d', 'fno'), f'Model name {model_name} invalid'
    if model_name == 'unet2d': 
        model = UNet2d(in_channels=in_channels,
                       out_channels=out_channels).cuda().float()
    elif model_name == 'fno':
        model = FNO(n_modes=(16, 16),
                    hidden_channels=32,
                    in_channels=in_channels,
                    out_channels=out_channels).cuda().float()
    print(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.optimizer.initial_lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=cfg.lr_scheduler.patience,
                                                   gamma=cfg.lr_scheduler.factor)

    TrainerClass = trainer_map[cfg.torch_dataset_name]
    trainer = TrainerClass(model,
                           train_dataloader,
                           val_dataloader,
                           optimizer,
                           lr_scheduler,
                           writer,
                           cfg)
    print(trainer)
    trainer.train(cfg.train.max_epochs)

    ckpt_file = f'{model.__class__.__name__}_{cfg.torch_dataset_name}.pt'
    ckpt_root = Path.home() / f'crsp/ai4ts/afeeney/thermal_models/{cfg.dataset.name}'
    Path(ckpt_root).mkdir(parents=True, exist_ok=True)
    ckpt_path = f'{ckpt_root}/{ckpt_file}'
    print(f'saving model to {ckpt_path}')
    torch.save(model, f'{ckpt_path}')
    
if __name__ == '__main__':
    train_app()
