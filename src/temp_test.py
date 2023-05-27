from omegaconf import DictConfig, OmegaConf
import hydra
from hdf5_dataset import HDF5Dataset, TempDataset
import torch
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import subprocess
from metrics import compute_metrics

@hydra.main(version_base=None, config_path='../conf', config_name='default')
def train_app(cfg):
    print(OmegaConf.to_yaml(cfg))

    writer = SummaryWriter(log_dir=cfg.log.log_dir)

    model = torch.load(Path.home() / 'crsp/ai4ts/afeeney/temp_model.pt').cuda().float()
    val_dataset = ConcatDataset([TempDataset(p, time_window=cfg.train.time_window) for p in cfg.dataset.val_paths])

    validate(model, val_dataset, writer)

def validate(model, dataset, writer):
    model.eval()

    temp_preds = []
    labels = []

    for timestep in range(len(dataset)):
        input, label = dataset[timestep]
        input = input.cuda().float().unsqueeze(0)
        label = label.cuda().float().unsqueeze(0)
        print(input.size())
        with torch.no_grad():
            pred = model(input)
            temp = pred[:, 0]
            temp_preds.append(temp)
            labels.append(label[:, 0])
    temps = torch.cat(temp_preds, dim=0)
    labels = torch.cat(labels, dim=0)

    metrics = compute_metrics(temps, labels)
    print(metrics)

    for i in range(temps.size(0)):
        i_str = str(i).zfill(3)
        pred_arr = temps[i].detach().cpu().numpy()
        plt.imsave(f'test_im/temp_pred_{i_str}.png', np.flipud(pred_arr), dpi=300)
        label_arr = labels[i].detach().cpu().numpy()
        plt.imsave(f'test_im/temp_label_{i_str}.png', np.flipud(label_arr), dpi=300)

        def plt_temp_arr(f, ax, arr, title):
            cm_object = ax.imshow(arr, vmin=0, vmax=1)
            ax.title.set_text(title)
            ax.axis('off')
            return cm_object

        f, axarr = plt.subplots(1, 2, layout="constrained")
        cm_object = plt_temp_arr(f, axarr[0], np.flipud(label_arr), 'Ground Truth Temperature')
        plt_temp_arr(f, axarr[1], np.flipud(pred_arr), 'Predicted Temperature')
        f.colorbar(cm_object, ax=axarr)
        plt.savefig(f'test_im/comp_{i_str}.png', dpi=300)
        plt.close()

if __name__ == '__main__':
    train_app()
