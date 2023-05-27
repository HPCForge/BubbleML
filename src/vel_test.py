from omegaconf import DictConfig, OmegaConf
import hydra
from hdf5_dataset import VelDataset
import torch
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
    model = torch.load(Path.home() / 'crsp/ai4ts/afeeney/thermal_models/wall_super_heat/FNO_vel_dataset.pt').cuda().float()
    val_dataset = VelDataset(cfg.dataset.val_paths[0], time_window=cfg.train.time_window)
    validate(model, val_dataset, writer)

def validate(model, dataset, writer):
    model.eval()

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
            pred = model(input)
            print(pred.shape)
            velx = pred[:, 0]
            vely = pred[:, 1]
            dataset.write_velx(velx, timestep)
            dataset.write_vely(vely, timestep)
            velx_preds.append(velx)
            vely_preds.append(vely)
            velx_labels.append(label[:, 0])
            vely_labels.append(label[:, 1])
    
    def print_metrics(pred, label):
        preds = torch.cat(pred, dim=0)
        labels = torch.cat(label, dim=0)
        metrics = compute_metrics(preds, labels)
        print(metrics)

    print('velx metrics:')
    print_metrics(velx_preds, velx_labels)
    print('vely metrics:')
    print_metrics(vely_preds, vely_labels)
    
    def mag(velx, vely):
        return np.sqrt(velx**2 + vely**2)
    
    vx_stack = torch.stack(velx_labels, dim=0)
    vy_stack = torch.stack(vely_labels, dim=0)

    gt = torch.max(torch.sqrt(vx_stack**2 + vy_stack**2)).cpu().detach().item()

    for i in range(len(velx_preds)):
        i_str = str(i).zfill(3)
        predx_arr = velx_preds[i].detach().cpu().squeeze(0).numpy()
        predy_arr = vely_preds[i].detach().cpu().squeeze(0).numpy()
        #plt.imsave(f'test_im/vel_pred_{i_str}.png', np.flipud(mag(predx_arr, predy_arr)), dpi=400)
        labelx_arr = velx_labels[i].detach().cpu().squeeze(0).numpy()
        labely_arr = vely_labels[i].detach().cpu().squeeze(0).numpy()
        #plt.imsave(f'test_im/vel_label_{i_str}.png', np.flipud(mag(labelx_arr, labely_arr)), dpi=400)

        def plt_temp_arr(f, ax, arr, mm, title):
            cm_object = ax.imshow(arr, vmin=0, vmax=mm, cmap='viridis')
            ax.title.set_text(title)
            ax.axis('off')
            return cm_object

        f, axarr = plt.subplots(2, 2, layout="constrained")
        label_mag = mag(labelx_arr, labely_arr)
        pred_mag = mag(predx_arr, predy_arr)

        cm_object = plt_temp_arr(f, axarr[0, 0], np.flipud(label_mag), gt, 'Ground Truth Temperature')
        plt_temp_arr(f, axarr[0, 1], np.flipud(pred_mag), gt, 'Predicted Temperature')
        f.colorbar(cm_object, ax=axarr[0, 1], fraction=0.05)

        cm_object = plt_temp_arr(f, axarr[1, 0], np.flipud(np.abs(label_mag - pred_mag)), 1, 'Absolute Error')
        f.colorbar(cm_object, ax=axarr[1, 0], fraction=0.05)

        xd = (labelx_arr - predx_arr) ** 2
        yd = (labely_arr - predy_arr) ** 2
        n = np.sqrt(xd + yd)
        cm_object = plt_temp_arr(f, axarr[1, 1], np.flipud(n), 1, 'L2 Error')
        f.colorbar(cm_object, ax=axarr[1, 1], fraction=0.05)

        plt.savefig(f'test_im/comp_{i_str}.png', dpi=500)
        plt.close()

if __name__ == '__main__':
    train_app()
