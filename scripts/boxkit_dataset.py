import glob
import boxkit
import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset
from pathlib import Path
import matplotlib.pyplot as plt
import h5py

class BoilingDataset(Dataset):
    def __init__(self, directory):
        super().__init__()
        filenames = sorted(glob.glob(directory + '/*'))
        self._filenames = [f for f in filenames if 'plt_cnt' in f]
        if len(self._filenames) > 0:
            self._data, self._params = self._load_data()
            self._load_dims()

    def to_hdf5(self, filename):
        if len(self._filenames) == 0:
            return
        with h5py.File(filename, 'w') as f:
            f.create_dataset('temperature', data=self._data['temp'])
            f.create_dataset('velx', data=self._data['velx'])
            f.create_dataset('vely', data=self._data['vely'])
            f.create_dataset('dfun', data=self._data['dfun'])
            f.create_dataset('pressure', data=self._data['pres'])
            f.create_dataset('x', data=self._data['x'])
            f.create_dataset('y', data=self._data['y'])

    def __len__(self):
        # len is the number of timesteps.
        # subtract 1 because the last timestep can
        # only be used as a label
        return self._data['temp'].size(2) - 1 

    def __getitem__(self, timestep):
        # TODO: use velx/vely input, only predict temp?
        # predict all 3? Include velx/vely at t+1 in input?
        temp_at_t = self._data['temp'][..., timestep]
        velx_at_t = self._data['velx'][..., timestep]
        vely_at_t = self._data['vely'][..., timestep]

        temp_next = self._data['temp'][..., timestep + 1]
        velx_next = self._data['velx'][..., timestep + 1]
        vely_next = self._data['vely'][..., timestep + 1]

        input = torch.stack([temp_at_t, velx_at_t, vely_at_t], dim=0)
        label = torch.stack([temp_next, velx_next, vely_next], dim=0)
        return input, label

    def inv_reynolds(self):
        return float(self._params['ins_invreynolds'])

    def prandtl(self):
        return float(self._params['ht_prandtl'])

    def alpha_vapor(self):
        r""" Computes the thermal diffusivity with mph_cpgas, mph_rhogas, mph_thcogas
        https://en.wikipedia.org/wiki/Thermal_diffusivity
        """
        cp = float(self._params['mph_cpgas'])
        rho = float(self._params['mph_rhogas'])
        thco = float(self._params['mph_thcogas'])
        return thco / (cp * rho)

    def _load_data(self):
        frame_dicts = self._load_files()
        var_dict = self._stack_frame_dicts(frame_dicts)
        params = self._load_params()
        return var_dict, params

    def _stack_frame_dicts(self, frame_dicts):
        var_list = frame_dicts[0].keys()
        var_dict = dict((v, []) for v in var_list)
        for frame in frame_dicts:
            for var in var_list:
                var_dict[var].append(torch.from_numpy(frame[var]))
        for var in var_list:
            var_dict[var] = torch.stack(var_dict[var], -1)
        return var_dict

    def _load_params(self):
        with h5py.File(self._filenames[0], 'r') as f:
            rrp = dict([(key.decode('utf-8').strip(), val) for (key, val) in f['real runtime parameters']])
        return rrp

    def _load_dims(self):
        frame0 = boxkit.read_dataset(self._filenames[0], source='flash')
        self.xmin, self.xmax = frame0.xmin, frame0.xmax
        self.ymin, self.ymax = frame0.ymin, frame0.ymax

    def _load_files(self):
        frames = []
        for filename in self._filenames:
            frame = boxkit.read_dataset(filename, source='flash')
            blocks = frame.blocklist
            y_bs, x_bs = frame.nyb, frame.nxb

            nblockx = int((frame.xmax - frame.xmin) / blocks[0].dx / x_bs)
            nblocky = int((frame.ymax - frame.ymin) / blocks[0].dy / y_bs)

            nxb = nblockx * x_bs
            nyb = nblocky * y_bs

            var_dict = {}
            for key in frame.varlist:
                var_dict[key] = np.empty((nxb, nyb))
                for block in blocks:
                    a, b, _ = block.get_relative_loc([frame.xmin, frame.ymin, 0])
                    r, c = b * y_bs, a * x_bs 
                    var_dict[key][r:r+y_bs, c:c+x_bs] = block[key]

            var_dict['x'] = np.empty((nxb, nyb))
            var_dict['y'] = np.empty((nxb, nyb))
            block_idx = 0
            for block in blocks:
                x, y = np.meshgrid(block.xrange('center'),
                                   block.yrange('center'))
                a, b, _ = block.get_relative_loc([frame.xmin, frame.ymin, 0])
                r, c = b * y_bs, a * x_bs 
                var_dict['x'][r:r+y_bs,c:c+x_bs] = x 
                var_dict['y'][r:r+y_bs,c:c+x_bs] = y

            print(var_dict)

            frames.append(var_dict)
        return frames

if __name__ == '__main__':
    target = str(Path.home() / 'crsp/ai4ts/share/PB_simulation/SubCooled-FC72-2D_HDF5/')
    base = str(Path.home() / 'crsp/ai4ts/share/PB_simulation/SubCooled-FC72-2D/')
    subdirs = glob.glob(f'{base}/*')

    for subdir in subdirs[:1]:
        print(subdir)
        b = BoilingDataset(subdir)
        dir_name = subdir[subdir.find('Twall-'):]
        b.to_hdf5(f'{target}/{dir_name}.hdf5')

    print('done!')