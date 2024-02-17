import glob
import boxkit
import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset
from pathlib import Path
import matplotlib.pyplot as plt
import h5py
from joblib import Parallel, delayed
import sys

class BoilingDataset(Dataset):
    def __init__(self, directory):
        super().__init__()
        filenames = sorted(glob.glob(directory + '/*'))
        self._filenames = [f for f in filenames if 'plt_cnt' in f][:-1]
        self.heater = [f for f in filenames if 'htr' in f][0]
        
        with h5py.File(self._filenames[0]) as f:
            print(f.keys())
        if len(self._filenames) > 0:
            self._data = self._load_data()
            self._load_dims()
        self._data['liquid_iters'] = torch.zeros(self._data['site_dfun'].shape)
        for idx, row in enumerate(self._data['site_dfun']):
            for jdx, val in enumerate(row):
                if val < 0:
                    self._data['liquid_iters'][idx, jdx] += 1
                else:
                    self._data['liquid_iters'][idx, jdx] = 0

    def to_hdf5(self, filename):
        if len(self._filenames) == 0:
            return
        print(filename)
        perm = (2, 0, 1)
        with h5py.File(filename, 'w') as f:
            f.create_dataset('temperature', data=self._data['temp'].permute(perm))
            f.create_dataset('velx', data=self._data['velx'].permute(perm))
            f.create_dataset('vely', data=self._data['vely'].permute(perm))
            f.create_dataset('dfun', data=self._data['dfun'].permute(perm))
            f.create_dataset('pressure', data=self._data['pres'].permute(perm))
            f.create_dataset('massflux', data=self._data['mflx'].permute(perm))
            f.create_dataset('normx', data=self._data['nrmx'].permute(perm))
            f.create_dataset('normy', data=self._data['nrmy'].permute(perm))
            f.create_dataset('x', data=self._data['x'].permute(perm))
            f.create_dataset('y', data=self._data['y'].permute(perm))
            f.create_dataset('x_sites', data=self._data['x_sites'])
            f.create_dataset('y_sites', data=self._data['y_sites'])
            f.create_dataset('site_dfun', data=self._data['site_dfun'].permute(1,0))
            f.create_dataset('liquid_iters', data=self._data['liquid_iters'].permute(1,0))
            
            attributes = {}
            with h5py.File(self._filenames[0], 'r') as plot_0:
                real_runtime_params = plot_0['real runtime parameters'][:]
                int_runtime_params = plot_0['integer runtime parameters'][:]
                attributes['real-runtime-params'] = {real_runtime_params[i][0].decode('utf-8').strip(): real_runtime_params[i][1] for i in range(real_runtime_params.shape[0])}
                attributes['int-runtime-params'] = {int_runtime_params[i][0].decode('utf-8').strip(): int_runtime_params[i][1] for i in range(int_runtime_params.shape[0])}
                heater = h5py.File(self.heater, 'r')
                attributes['heater'] = {k: heater['heater'][k][...] for k in heater['heater'].keys()}
                attributes['heater']['nucSeedRadius'] = heater['init']['radii'][...][0]
                f.attrs.update(attributes)


    def _load_data(self):
        frame_dicts = self._load_files_par()
        var_dict = self._stack_frame_dicts(frame_dicts)
        return var_dict

    def _stack_frame_dicts(self, frame_dicts):
        var_list = frame_dicts[0].keys()
        var_dict = dict((v, []) for v in var_list)
        for frame in frame_dicts:
            for var in var_list:
                var_dict[var].append(torch.from_numpy(frame[var]))
        for var in var_list:
            var_dict[var] = torch.stack(var_dict[var], -1)
        return var_dict

    def _load_dims(self):
        frame0 = boxkit.read_dataset(self._filenames[0], source='flash')
        self.xmin, self.xmax = frame0.xmin, frame0.xmax
        self.ymin, self.ymax = frame0.ymin, frame0.ymax

    def _load_files_par(self):
        NJOBS = 30
        output = Parallel(n_jobs=NJOBS)(delayed(self._load_file)(idx, filename) for idx, filename in enumerate(self._filenames))
        return output

    def _load_file(self, idx, filename):
        frame = boxkit.read_dataset(filename, source='flash')
        blocks = frame.blocklist
        y_bs, x_bs = frame.nyb, frame.nxb

        blockx_pixel = x_bs * round(int((frame.xmax - frame.xmin)/blocks[0].dx)/x_bs)
        blocky_pixel = y_bs * round(int((frame.ymax - frame.ymin)/blocks[0].dy)/y_bs)
        nblockx = int(blockx_pixel/ x_bs)
        nblocky = int(blocky_pixel/ y_bs)

        nxb = nblockx * x_bs
        nyb = nblocky * y_bs

        var_dict = {}
        for key in frame.varlist:
            var_dict[key] = np.zeros((nyb, nxb))
            for block in blocks:
                r = y_bs * round(int((nyb * (block.ymin - frame.ymin))/(frame.ymax - frame.ymin))/y_bs)
                c = x_bs * round(int((nxb * (block.xmin - frame.xmin))/(frame.xmax - frame.xmin))/x_bs)
                var_dict[key][r:r+y_bs, c:c+x_bs] = block[key]

        var_dict['x'] = np.zeros((nyb, nxb))
        var_dict['y'] = np.zeros((nyb, nxb))

        for block in blocks:
            x, y = np.meshgrid(block.xrange('center'),
                               block.yrange('center'))
            r = y_bs * round(int((nyb * (block.ymin - frame.ymin))/(frame.ymax - frame.ymin))/y_bs)
            c = x_bs * round(int((nxb * (block.xmin - frame.xmin))/(frame.xmax - frame.xmin))/x_bs)
            var_dict['x'][r:r+y_bs,c:c+x_bs] = x 
            var_dict['y'][r:r+y_bs,c:c+x_bs] = y

        coordx, coordy = var_dict['x'][0], np.transpose(var_dict['y'])[0]
        heater = h5py.File(self.heater, 'r')
        heater_sites = list(zip(heater['site']['x'][...], heater['site']['y'][...]))
        var_dict['x_sites'] = heater['site']['x'][...]
        var_dict['y_sites'] = heater['site']['y'][...]
        var_dict['site_dfun'] = np.zeros(len(heater_sites))

        seed_height = heater['init']['radii'][...][0] * np.cos(heater['heater']['rcdAngle'][...][0] * (np.pi/180))
        for i, htr_points_xy in enumerate(heater_sites):
            seed_x = htr_points_xy[0]
            seed_y = htr_points_xy[1] + seed_height

            x_i = np.searchsorted(coordx, seed_x, side='left')
            y_i = np.searchsorted(coordy, seed_y, side='left')
            var_dict['site_dfun'][i] = (var_dict['dfun'][y_i, x_i] + var_dict['dfun'][y_i-1, x_i] + var_dict['dfun'][y_i, x_i-1] + var_dict['dfun'][y_i-1, x_i-1])/4.0

        return var_dict

TWALL = 'Twall-'

def unblock_dataset(write_dir, read_dir):
    b = BoilingDataset(read_dir)

    filename = Path(read_dir).stem
    print(filename)
    assert TWALL in filename, f'eek {TWALL} not in filename'
    wall_temp = int(filename[len(TWALL):])
    print(wall_temp)

    dir_name = read_dir[read_dir.find(TWALL):]
    b.to_hdf5(f'{target}/{dir_name}.hdf5')

if __name__ == '__main__':
    target = str(Path.home() / '/share/crsp/lab/ai4ts/share/BubbleML-2.0/PoolBoiling-Saturated-FC72-2D-0.1/')
    Path(target).mkdir(parents=True, exist_ok=True)

    base = str(Path.home() / '/pub/sheikhh1/bubbleml/BubbleML-2.0/simulation/PoolBoiling-Saturated-FC72-2D/')

    subdirs = [f for f in glob.glob(f'{base}/*') if TWALL in f]
    print(subdirs)
    
    for idx, subdir in enumerate(subdirs):
        print(f'processing {subdir} {idx}/{len(subdirs)}')
        unblock_dataset(target, subdir)
    
    print('done!')
