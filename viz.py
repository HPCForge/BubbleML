import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import h5py
import argparse
import numpy as np
import pathlib

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, help='path to hdf5 file to visualize')
    args = parser.parse_args()

    with h5py.File(args.file) as f:
        temp = f['temperature'][:]
        pres = f['pressure'][:]
        velx = f['velx'][:]
        vely = f['vely'][:]
        pos = f['dfun'][:]
        pos[pos <= 0] = 0

    def magnitude(x, y):
        return np.sqrt(x**2 + y**2)

    mag = magnitude(velx, vely)
    print(mag.min(), mag.max())

    plt_arr(temp, temp_cmap(), 'temp', vmax=1, ticks=[0, 0.2, 0.6, 0.9])
    plt_arr(mag, 'jet', 'vel', vmax=12, ticks=np.linspace(0, 12, 3))
    plt_arr(pos, 'Purples', 'pos', vmax=0.5, ticks=[0, 0.5], use_colorbar=False)

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
    
def plt_arr(frames, cmap, name, vmax, ticks, use_colorbar=True):
    plt.rc("font", family="serif", size=12, weight="bold")
    plt.rc("axes", labelweight="bold")
    for i in range(frames.shape[2]):
        i_str = str(i).zfill(3)

        def plt_temp_arr(f, ax, arr):
            cm_object = ax.imshow(arr, vmin=0, vmax=vmax, cmap=cmap)
            ax.axis('off')
            return cm_object

        arr = frames[..., i]
        f, axarr = plt.subplots(1, 1)
        cm_object = plt_temp_arr(f, axarr, np.flipud(arr))

        f.tight_layout()
        if use_colorbar:
            f.colorbar(cm_object,
                       #ax=axarr.ravel().tolist(),
                       ticks=ticks,
                       fraction=0.01,
                       pad=0.02)

        f.set_size_inches(w=6, h=3)
        pathlib.Path(f'test_im/{name}').mkdir(parents=True, exist_ok=True)
        plt.savefig(f'test_im/{name}/{i_str}.png',
                    dpi=600,
                    bbox_inches='tight',
                    transparent=True)
        plt.close()

if __name__ == '__main__':
    main()
