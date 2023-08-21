import argparse
import os
import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt

L_C = 0.7 # The characteristic length of the fluid in mm. Multiply the non-dimensional length and height of the domain to get real world dimensions in mm.
U_C = 82.867 # The characteristic velocity of the fluid in mm/s. Multiply the non-dimensional velocities to get real world velocities in mm/s
T_C = 0.008 # The characteristic time of the fluid in s. Multiply with non-dimensional time to get the dimensional time.

def write_flo(file_path, u, v):
    """ 
    Write optical flow to file.
    :param file_path: Path obtained from user to write optical flow file
    :param u: np.ndarray is assumed to contain u channel or x velocities,
    :param v: np.ndarray is assumed to contain v channel or y velocities,
    """
    nBands = 2

    assert (u.shape == v.shape)
    height, width = u.shape
    f = open(file_path, 'wb')
    # write the header
    np.array([202021.25]).astype(np.float32).tofile(f)
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)
    # arrange into matrix form
    tmp = np.zeros((height, width * nBands))
    tmp[:, np.arange(width) * 2] = u
    tmp[:, np.arange(width) * 2 + 1] = v
    tmp.astype(np.float32).tofile(f)

    f.close()

def make_dataset(sim_file, op_dir, train_valid_split, plot_interval):
    simul_data = h5py.File(sim_file, 'r')
    dist_fields = simul_data['dfun'][:]
    num_timesteps = dist_fields.shape[0]
    train_length = int(train_valid_split * num_timesteps)
    train_save_dir = os.path.join(op_dir, 'train', sim_file.split('/')[-1][:-5])
    valid_save_dir = os.path.join(op_dir, 'valid', sim_file.split('/')[-1][:-5])
    
    domain_height = round(L_C * (simul_data['y'][-1,-1,-1] + simul_data['y'][0,0,0]), 2)
    pixel_height = simul_data['y'].shape[1]
    pixel_density = pixel_height/domain_height
    
    secs_per_frame = T_C * plot_interval
    
    os.makedirs(os.path.join(train_save_dir, 'img'), exist_ok=True)
    os.makedirs(os.path.join(train_save_dir, 'flow'), exist_ok=True)
    os.makedirs(os.path.join(valid_save_dir, 'img'), exist_ok=True)
    os.makedirs(os.path.join(valid_save_dir, 'flow'), exist_ok=True)

    for index in range(train_length):
        dist_field = np.flipud(simul_data['dfun'][()][index,:,:])
        u = np.flipud(simul_data['velx'][()][index,:,:])
        v = -1*np.flipud(simul_data['vely'][()][index,:,:])
        u[dist_field < 0] = 0
        v[dist_field < 0] = 0
        u = u * U_C * pixel_density * secs_per_frame
        v = v * U_C * pixel_density * secs_per_frame 
        dist_field[dist_field>0] *= (255/dist_field.max())
        dist_field[dist_field<0] = 255
        dist_field = dist_field.astype(np.uint8)
        plt.imsave(os.path.join(train_save_dir, 'img', '{:04d}'.format(index) + '.png'), dist_field, cmap='gray')
        write_flo(os.path.join(train_save_dir, 'flow', '{:04d}'.format(index) + '.flo'), u, v)
        if index%10 == 0:
            print(f'{index} files done for {sim_file}, u_max = {u.max()}, v_max = {v.max()}')

    for index in range(train_length, num_timesteps):
        dist_field = np.flipud(simul_data['dfun'][()][index,:,:])
        u = np.flipud(simul_data['velx'][()][index,:,:])
        v = -1*np.flipud(simul_data['vely'][()][index,:,:])
        u[dist_field < 0] = 0
        v[dist_field < 0] = 0
        u = u * U_C * pixel_density * secs_per_frame
        v = v * U_C * pixel_density * secs_per_frame
        dist_field[dist_field>0] *= (255/dist_field.max())
        dist_field[dist_field<0] = 255
        dist_field = dist_field.astype(np.uint8)
        plt.imsave(os.path.join(valid_save_dir, 'img', '{:04d}'.format(index) + '.png'), dist_field, cmap='gray')
        write_flo(os.path.join(valid_save_dir, 'flow', '{:04d}'.format(index) + '.flo'), u, v)
        if index%10 == 0:
            print(f'{index} files done for {sim_file}, u_max = {u.max()}, v_max = {v.max()}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--ip_dir', type=str, 
                        help='path to the directory of hdf5 simulation files of a particular boiling study')
    parser.add_argument('--op_dir', type=str, default='Boiling', 
                        help='path to save the optical flow training set')
    parser.add_argument('--train_valid_split', type=float, default=0.8, 
                        help='percentage of images to be kept for validation')
    parser.add_argument('--plot_interval', type=float, default=1.0,
                        help='non-dimensional time interval at which simulation plot files were generated.')
    args = parser.parse_args()

    sim_files = glob.glob(f'{args.ip_dir}/*.hdf5')

    for sim_file in sim_files:
        make_dataset(sim_file, args.op_dir, args.train_valid_split, args.plot_interval)
