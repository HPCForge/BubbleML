import argparse
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
import numpy as np
import os
from pathlib import Path
import subprocess
import scipy.fft as sfft
from dataclasses import dataclass

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', required=True, type=str,
                        help='Path to directory with model and sim output.pt files')
    return parser.parse_args()

@dataclass
class BoilingData:
    temp: torch.Tensor

def load_vel_data(temp_path):
    pred = BoilingData(
            torch.load(f'{temp_path}/model_ouput.pt').numpy())
    label = BoilingData(
            torch.load(f'{temp_path}/sim_ouput.pt').numpy())
    return pred, label

def main():
    args = parse_args()
    
    job_id = '25032868/'
    pred, label = load_vel_data(f'test_im/temp/{job_id}')
    
    plt_temp(pred.temp, label.temp, args.path, 'model')

    subprocess.call(
            f'ffmpeg -y -framerate 25 -pattern_type glob -i "{args.path}/*.png" output.mp4',
            shell=True)

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

def fft(x):
    x_fft = sfft.fft2(x)
    x_shift = np.abs(sfft.fftshift(x_fft))
    return x_shift

def mag(velx, vely):
    return np.sqrt(velx**2 + vely**2)

def plt_vel(pred, label, path, model_name):
    plt.rc("font", family="serif", size=16, weight="bold")
    plt.rc("axes", labelweight="bold")

    label_mag = mag(label.velx, label.vely)
    pred_mag = mag(pred.velx, pred.vely)
    mag_vmax = abs(pred_mag[:50]).max()
    print(label_mag.max(), pred_mag.max())

    frames = min(pred.temp.shape[0], 100)
    for i in range(frames):
        i_str = str(i).zfill(3)
        f, ax = plt.subplots(2, 2, layout='constrained')
        
        #x_vmax, x_vmin = label.velx.max(), label.velx.min()
        #y_vmax, y_vmin = label.vely.max(), label.vely.min()

        cm_object = ax[0, 0].imshow(np.flipud(label.temp[i]), vmin=0, vmax=1, cmap=temp_cmap())
        #ax[1, 0].imshow(np.flipud(label.velx[i]), vmin=x_vmin, vmax=x_vmax, cmap='jet')
        #ax[2, 0].imshow(np.flipud(label.vely[i]), vmin=y_vmin, vmax=y_vmax, cmap='jet')
        #ax[1, 0].imshow(np.flipud(label_mag[i]), vmin=0, vmax=mag_vmax, cmap='jet')

        ax[0, 1].imshow(np.flipud(np.nan_to_num(pred.temp[i])), vmin=0, vmax=1, cmap=temp_cmap())
        #ax[1, 1].imshow(np.flipud(pred.velx[i]), vmin=x_vmin, vmax=x_vmax, cmap='jet')
        #ax[2, 1].imshow(np.flipud(pred.vely[i]), vmin=y_vmin, vmax=x_vmax, cmap='jet')
        #ax[1, 1].imshow(np.flipud(pred_mag[i]), vmin=0, vmax=mag_vmax, cmap='jet')

        ax[0, 0].axis('off')
        ax[1, 0].axis('off')
        ax[0, 1].axis('off')
        ax[1, 1].axis('off')

        #ax[0, 2].imshow(np.flipud(fft(label.temp[i])))
        #ax[1, 2].imshow(np.flipud(fft(label.velx[i])))
        #ax[2, 2].imshow(np.flipud(fft(label.vely[i])))
        #ax[3, 2].imshow(np.flipud(fft(label_mag)))

        #ax[0, 3].imshow(np.flipud(fft(pred.temp[i])))
        #ax[1, 3].imshow(np.flipud(fft(pred.velx[i])))
        #ax[2, 3].imshow(np.flipud(fft(pred.vely[i])))
        #ax[3, 3].imshow(np.flipud(fft(pred_mag)))

        im_path = Path(path)
        im_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(f'{str(im_path)}/{i_str}.png',
                    dpi=200,
                    bbox_inches='tight',
                    transparent=True)
        plt.close()
    

def plt_temp(temps, labels, path, model_name):
    print(temps.min(), temps.max(),
          labels.min(), labels.max())

    plt.rc("font", family="serif", size=16, weight="bold")
    plt.rc("axes", labelweight="bold")
    for i in range(len(temps)):
        i_str = str(i).zfill(3)

        def plt_temp_arr(f, ax, arr, title):
            cm_object = ax.imshow(arr, vmin=0, vmax=1, cmap=temp_cmap())
            #ax.set_title(title)
            ax.axis('off')
            return cm_object

        temp = temps[i]
        label = labels[i]
        f, axarr = plt.subplots(2, 3, layout="constrained")
        cm_object = plt_temp_arr(f, axarr[0, 0], np.flipud(label), 'Ground Truth')
        cm_object = plt_temp_arr(f, axarr[0, 1], np.flipud(temp), model_name)
        
        err = np.abs(temp - label)
        cm_object = plt_temp_arr(f, axarr[0, 2], np.flipud(err), 'Absolute Error')
        f.tight_layout()
        f.colorbar(cm_object,
                   ax=axarr.ravel().tolist(),
                   ticks=[0, 0.2, 0.6, 0.9],
                   fraction=0.04,
                   pad=0.02)
        f.set_size_inches(w=6, h=3)
        
        label_h = fft(label)
        temp_h = fft(temp)
        err_h = np.abs(label_h - temp_h)

        axarr[1, 0].imshow(np.flipud(label_h))
        axarr[1, 1].imshow(np.flipud(temp_h))
        axarr[1, 2].imshow(np.flipud(err_h))

        im_path = Path(path)
        im_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(f'{str(im_path)}/{i_str}.png',
                    dpi=600,
                    bbox_inches='tight',
                    transparent=True)
        plt.close()

if __name__ == '__main__':
    main()
