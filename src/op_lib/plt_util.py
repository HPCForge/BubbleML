import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import torch

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
    with open('maes', 'w+') as f:
        for rmse in rmses:
            f.write(f'{rmse}\n')
    plt.plot(range(len(rmses)), rmses)
    plt.xlabel('Forward Prop Iteration')
    plt.ylabel('RMSE')
    plt.savefig('iter_mae.png', bbox_inches='tight', dpi=500)

def plt_temp(temps, labels, model_name):
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
        f, axarr = plt.subplots(1, 2, layout="constrained")
        cm_object = plt_temp_arr(f, axarr[0], np.flipud(label), 'Ground Truth')
        plt_temp_arr(f, axarr[1], np.flipud(temp), model_name)
        
        err = np.abs(temp - label)
        #cm_object = plt_temp_arr(f, axarr[2], np.flipud(err), 'Absolute Error')
        f.colorbar(cm_object,
                   ax=axarr.ravel().tolist(),
                   ticks=[0, 0.2, 0.6, 0.9],
                   fraction=0.02,
                   pad=0.02)

        f.set_size_inches(w=6, h=3)
        plt.savefig(f'test_im/temp/{i_str}.png', dpi=600, transparent=True)
        plt.close()

def plt_vel(vel_preds, vel_labels, max_mag, model_name):
    for i in range(len(vel_preds)):
        i_str = str(i).zfill(3)

        def plt_temp_arr(f, ax, arr, mm, title):
            cm_object = ax.imshow(arr, vmin=0, vmax=mm, cmap='viridis')
            ax.set_title(title)
            ax.axis('off')
            return cm_object

        f, axarr = plt.subplots(1, 3, layout="constrained")
        pred_mag = vel_preds[i]
        label_mag = vel_labels[i] 

        cm_object = plt_temp_arr(f, axarr[0], np.flipud(label_mag), max_mag, 'Ground Truth')
        plt_temp_arr(f, axarr[1], np.flipud(pred_mag), max_mag, model_name)
        f.colorbar(cm_object, ax=axarr[1], fraction=0.05)

        cm_object = plt_temp_arr(f, axarr[2], np.flipud(np.abs(label_mag - pred_mag)), 10, 'Absolute Error')
        f.colorbar(cm_object, ax=axarr[2], fraction=0.05)

        #vx_pred, vx_label = velx_preds[i], velx_labels[i]
        #vy_pred, vy_label = vely_preds[i], vely_labels[i]
        #xd = (vx_pred - vx_label) ** 2
        #yd = (vy_pred - vy_label) ** 2
        #n = np.sqrt(xd + yd)
        #cm_object = plt_temp_arr(f, axarr[1, 1], np.flipud(n), 1, 'L2 Error')
        #f.colorbar(cm_object, ax=axarr[1, 1], fraction=0.05)

        plt.savefig(f'test_im/vel/{i_str}.png', dpi=500)
        plt.close()
