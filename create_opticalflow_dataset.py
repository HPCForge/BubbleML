import argparse
import os
import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt

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

def make_dataset(sim_file, op_dir, train_valid_split):
    simul_data = h5py.File(sim_file, 'r')
    num_timesteps = simul_data['dfun'][()].shape[2]
    train_length = int(train_valid_split * num_timesteps)
    train_save_dir = os.path.join(op_dir, 'train', sim_file.split('/')[-1][:-5])
    valid_save_dir = os.path.join(op_dir, 'valid', sim_file.split('/')[-1][:-5])
    os.makedirs(os.path.join(train_save_dir, 'img'), exist_ok=True)
    os.makedirs(os.path.join(train_save_dir, 'flow'), exist_ok=True)
    os.makedirs(os.path.join(valid_save_dir, 'img'), exist_ok=True)
    os.makedirs(os.path.join(valid_save_dir, 'flow'), exist_ok=True)

    for index in range(train_length):
        dist_field = simul_data['dfun'][()][:,:,index]
        u = simul_data['velx'][()][:,:,index]
        v = simul_data['vely'][()][:,:,index]
        u[dist_field < 0] = 0
        v[dist_field < 0] = 0
        img_arr = np.zeros((dist_field.shape[0], dist_field.shape[1]), dtype=np.uint8)
        for i in range(dist_field.shape[0]):
            for j in range(dist_field.shape[1]):
                if dist_field[i][j] < 0:
                    img_arr[i][j] = 255
                else:
                    img_arr[i][j] = int((dist_field[i][j]/np.max(dist_field))*255)
        img_arr = np.flipud(img_arr)
        u = np.flipud(u)
        v = -1 * np.flipud(v)
        plt.imsave(os.path.join(train_save_dir, 'img', '{:04d}'.format(index) + '.png'), img_arr, cmap='gray')
        write_flo(os.path.join(train_save_dir, 'flow', '{:04d}'.format(index) + '.flo'), u, v)
        if index%10 == 0:
            print(f'{index} files done for {sim_file}')

    for index in range(train_length, num_timesteps):
        dist_field = simul_data['dfun'][()][:,:,index]
        u = simul_data['velx'][()][:,:,index]
        v = simul_data['vely'][()][:,:,index]
        u[dist_field < 0] = 0
        v[dist_field < 0] = 0
        img_arr = np.zeros((dist_field.shape[0], dist_field.shape[1]), dtype=np.uint8)
        for i in range(dist_field.shape[0]):
            for j in range(dist_field.shape[1]):
                if dist_field[i][j] < 0:
                    img_arr[i][j] = 255
                else:
                    img_arr[i][j] = int((dist_field[i][j]/np.max(dist_field))*255)
        img_arr = np.flipud(img_arr)
        u = np.flipud(u)
        v = -1 * np.flipud(v)
        plt.imsave(os.path.join(valid_save_dir, 'img', '{:04d}'.format(index) + '.png'), img_arr, cmap='gray')
        write_flo(os.path.join(valid_save_dir, 'flow', '{:04d}'.format(index) + '.flo'), u, v)
        if index%10 == 0:
            print(f'{index} files done for {sim_file}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--ip_dir', type=str, help='path to the directory of hdf5 simulation files of a particular boiling study. file extensions should be .hdf5')
    parser.add_argument('--op_dir', type=str, default='Boiling', help='path to save the optical flow training set')
    parser.add_argument('--train_valid_split', type=float, default=0.8, help='percentage of images to be kept for validation')
    
    args = parser.parse_args()

    sim_files = glob.glob(f'{args.ip_dir}/*.hdf5')

    for sim_file in sim_files:
        make_dataset(sim_file, args.op_dir, args.train_valid_split)
