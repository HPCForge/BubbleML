import argparse
import pathlib

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import h5py
import numpy as np
import cv2


def plot_arr(dist_fields, temp_fields, pres_fields, velx_fields, vely_fields, velmag_fields, mflux_fields, normx_fields, normy_fields, op_dir):
    """
    input: 3D array of variable frames (x, y, t)
    input: output directory
    Plots frames for each individual timestep
    """
    timesteps = dist_fields.shape[0]
    
    for i in range(timesteps):
        i_str = str(i).zfill(3)
        
        dist_field = np.copy(np.flipud(dist_fields[i, :, :]))
        temp_field = np.flipud(temp_fields[i, :, :])
        pres_field = np.flipud(pres_fields[i, :, :])
        velx_field = np.flipud(velx_fields[i, :, :])
        vely_field = -1 * np.flipud(vely_fields[i, :, :])
        velmag_field = np.flipud(velmag_fields[i, :, :])
        mflux_field = np.flipud(mflux_fields[i, :, :])
        normx_field = np.flipud(normx_fields[i, :, :])
        normy_field = np.flipud(normy_fields[i, :, :])
        
        # Plot distance field
        dist_field[dist_field>0] *= (255/dist_field.max())
        dist_field[dist_field<0] = 255
        dist_field = dist_field.astype(np.uint8)
        edge_map = cv2.Canny(dist_field, 0, 255)
        kernel = np.ones((3,3),np.uint8)
        edge_map = cv2.dilate(edge_map, kernel, iterations=1)
        mask = np.where(edge_map > 0, 0, 255)
        alpha = np.where(mask > 0, 0, 255)
        overlay = np.dstack((mask, mask, mask, alpha))
        pathlib.Path(f'{op_dir}/dist').mkdir(parents=True, exist_ok=True)
        plt.imsave(f'{op_dir}/dist/{i_str}.png', dist_field, cmap='GnBu')
        plt.close()

        # Plot temperature field
        temp_ranges = [0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.134, 0.167, 
                       0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        color_codes = ['#0000FF', '#0443FF', '#0E7AFF', '#16B4FF', '#1FF1FF', '#21FFD3',
                    '#22FF9B', '#22FF67', '#22FF15', '#29FF06', '#45FF07', '#6DFF08',
                    '#9EFF09', '#D4FF0A', '#FEF30A', '#FEB709', '#FD7D08', '#FC4908',
                    '#FC1407', '#FB0007']
        colors = list(zip(temp_ranges, color_codes))
        cmap = LinearSegmentedColormap.from_list('temperature_colormap', colors)
            
        fig, ax = plt.subplots()
        im = ax.imshow(temp_field, cmap=cmap)
        ax.imshow(overlay, alpha=1)

        ax.set_aspect('equal')
        ax.axis('off')
        fig.colorbar(im, ticks=[0, 0.2, 0.6, 0.9], fraction=0.04, pad=0.05, location='bottom')
        pathlib.Path(f'{op_dir}/temp').mkdir(parents=True, exist_ok=True)
        plt.savefig(f'{op_dir}/temp/{i_str}.png', bbox_inches='tight')
        plt.close()

        # Plot pressure field
        pres_field = (pres_field - pres_field.min())/(pres_field.max() - pres_field.min())
        fig, ax = plt.subplots()
        mask = np.where(edge_map > 0, 0, 255)
        alpha = np.where(mask > 0, 0, 255)
        overlay = np.dstack((mask, mask, mask, alpha))

        im = ax.imshow(pres_field, cmap='seismic')
        ax.imshow(overlay, alpha=1)

        ax.set_aspect('equal')
        ax.axis('off')
        fig.colorbar(im, fraction=0.04, pad=0.05, location='bottom')
        pathlib.Path(f'{op_dir}/pres').mkdir(parents=True, exist_ok=True)
        plt.savefig(f'{op_dir}/pres/{i_str}.png', bbox_inches='tight')
        plt.close()

        # Plot velocity field
        x = np.arange(0,velmag_field.shape[1],1)
        y = np.arange(0,velmag_field.shape[0],1)
        X,Y = np.meshgrid(x,y)

        velmag_field[np.flipud(dist_fields[i, :, :])<0] = 0
        velx_field[np.flipud(dist_fields[i, :, :])>0] = 0
        vely_field[np.flipud(dist_fields[i, :, :])>0] = 0

        fig, ax = plt.subplots()
        im = ax.imshow(velmag_field, vmin=0, vmax=3, cmap='Purples')
        ax.imshow(overlay, alpha=1)
        ax.streamplot(X,Y,velx_field,vely_field, density=1.5, color='red')
        ax.set_aspect('equal')
        ax.axis('off')
        fig.colorbar(im, fraction=0.04, pad=0.05, location='bottom')
        pathlib.Path(f'{op_dir}/vel').mkdir(parents=True, exist_ok=True)
        plt.savefig(f'{op_dir}/vel/{i_str}.png', bbox_inches='tight')
        plt.close()

        fig, ax = plt.subplots()
        q = ax.quiver(X[::4,::4], Y[::4,::4], normx_field[::4,::4], normy_field[::4,::4], scale=30)
        ax.imshow(overlay, alpha=1)
        ax.set_aspect('equal')
        ax.axis('off')
        pathlib.Path(f'{op_dir}/norm_vecs').mkdir(parents=True, exist_ok=True)
        plt.savefig(f'{op_dir}/norm_vecs/{i_str}.png', bbox_inches='tight')
        plt.close()
        
        fig, ax = plt.subplots()
        im = ax.imshow(mflux_field, vmax=0.005, vmin=-0.005, cmap='bwr')
        ax.set_aspect('equal')
        fig.colorbar(im, fraction=0.04, pad=0.05)
        pathlib.Path(f'{op_dir}/mflux').mkdir(parents=True, exist_ok=True)
        plt.savefig(f'{op_dir}/mflux/{i_str}.png', bbox_inches='tight')
        plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, help='path to hdf5 file to visualize')
    parser.add_argument('--output_dir', type=str, help='path to output directory')
    args = parser.parse_args()

    simul_file = h5py.File(args.file, "r")
    
    dist_fields = simul_file['dfun'][:]
    temp_fields = simul_file['temperature'][:]
    pres_fields = simul_file['pressure'][:]
    velx_fields = simul_file['velx'][:]
    vely_fields = simul_file['vely'][:]
    velmag_fields = np.sqrt(velx_fields**2 + vely_fields**2)
    mflux_fields = simul_file['massflux'][:]
    normx_fields = simul_file['normx'][:]
    normy_fields = simul_file['normy'][:]

    plot_arr(dist_fields, temp_fields, pres_fields, velx_fields, vely_fields, velmag_fields, mflux_fields, normx_fields, normy_fields, op_dir=args.output_dir)
