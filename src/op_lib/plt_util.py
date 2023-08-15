import os
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import torch
from pathlib import Path

def temp_cmap():
    temp_ranges = [0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.134, 0.167,
                    0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    color_codes = ['#0000FF', '#0443FF', '#0E7AFF', '#16B4FF', '#1FF1FF', '#21FFD3',
                   '#22FF9B', '#22FF67', '#22FF15', '#29FF06', '#45FF07', '#6DFF08',
                   '#9EFF09', '#D4FF0A', '#FEF30A', '#FEB709', '#FD7D08', '#FC4908',
                   '#FC1407', '#FB0007']
    colors = list(zip(temp_ranges, color_codes))
    cmap = LinearSegmentedColormap.from_list('temperature_colormap', colors)
    return cmap

def plt_iter_mae(temps, labels):
    plt.rc("font", family="serif", size=14, weight="bold")
    plt.rc("axes", labelweight="bold")
    rmses = []
    for i in range(len(temps)):
        rmse = torch.sqrt(torch.mean(((temps[i] - labels[i]) ** 2).detach().cpu()))
        rmses.append(rmse)
    job_id = os.environ['SLURM_JOB_ID']
    with open('test/temp/{job_id}/iter_rmses', 'w+') as f:
        for rmse in rmses:
            f.write(f'{rmse}\n')

def plt_temp(temps, labels, model_name):
    temps = (temps + 1) / 2
    labels = (labels + 1) / 2

    plt.rc("font", family="serif", size=16, weight="bold")
    plt.rc("axes", labelweight="bold")
    for i in range(len(temps)):
        i_str = str(i).zfill(3)

        def plt_temp_arr(f, ax, arr, title):
            cm_object = ax.imshow(arr, vmin=0, vmax=1, cmap=temp_cmap())
            #ax.set_title(title)
            ax.axis('off')
            return cm_object

        temp = temps[i].numpy()
        label = labels[i].numpy()
        f, axarr = plt.subplots(1, 3, layout="constrained")
        cm_object = plt_temp_arr(f, axarr[0], np.flipud(label), 'Ground Truth')
        cm_object = plt_temp_arr(f, axarr[1], np.flipud(temp), model_name)
        
        err = np.abs(temp - label)
        cm_object = plt_temp_arr(f, axarr[2], np.flipud(err), 'Absolute Error')
        f.tight_layout()
        f.colorbar(cm_object,
                   ax=axarr.ravel().tolist(),
                   ticks=[0, 0.2, 0.6, 0.9],
                   fraction=0.04,
                   pad=0.02)
        f.set_size_inches(w=6, h=3)

        job_id = os.environ['SLURM_JOB_ID']
        im_path = Path(f'test_im/temp/{job_id}/')
        im_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(f'{str(im_path)}/{i_str}.png',
                    dpi=400,
                    bbox_inches='tight',
                    transparent=True)
        plt.close()

        torch.save(temps, f'{im_path}/model_ouput.pt')
        torch.save(labels, f'{im_path}/sim_ouput.pt')

def plt_vel(vel_preds, vel_labels, max_mag, model_name):

    #vel_preds = (vel_preds + 1) / 2
    #vel_labels = (vel_labels + 1) / 2

    max_mag = vel_labels.max()
    min_mag = vel_labels.min()

    for i in range(len(vel_preds)):
        i_str = str(i).zfill(3)

        def plt_temp_arr(f, ax, arr, title):
            cm_object = ax.imshow(arr, vmin=0, vmax=0.3, cmap='jet')
            ax.set_title(title)
            ax.axis('off')
            return cm_object

        f, axarr = plt.subplots(1, 2, layout="constrained")
        pred_mag = vel_preds[i]
        label_mag = vel_labels[i] 

        cm_object = plt_temp_arr(f, axarr[0], np.flipud(label_mag), 'Ground Truth')
        plt_temp_arr(f, axarr[1], np.flipud(pred_mag), model_name)
        f.colorbar(cm_object, ax=axarr[1], fraction=0.05)

        #cm_object = plt_temp_arr(f, axarr[2], np.flipud(np.abs(label_mag - pred_mag)), 1, 'Absolute Error')
        #f.colorbar(cm_object, ax=axarr[2], fraction=0.05)

        #vx_pred, vx_label = velx_preds[i], velx_labels[i]
        #vy_pred, vy_label = vely_preds[i], vely_labels[i]
        #xd = (vx_pred - vx_label) ** 2
        #yd = (vy_pred - vy_label) ** 2
        #n = np.sqrt(xd + yd)
        #cm_object = plt_temp_arr(f, axarr[1, 1], np.flipud(n), 1, 'L2 Error')
        #f.colorbar(cm_object, ax=axarr[1, 1], fraction=0.05)

        job_id = os.environ['SLURM_JOB_ID']
        im_path = Path(f'test_im/vel/{job_id}/')
        im_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(f'{str(im_path)}/{i_str}.png', dpi=500)
        plt.close()

        torch.save(vel_preds, f'{im_path}/model_ouput.pt')
        torch.save(vel_labels, f'{im_path}/sim_ouput.pt')
