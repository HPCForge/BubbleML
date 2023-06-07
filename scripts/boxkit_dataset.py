import glob
import boxkit
import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset
from pathlib import Path
import matplotlib.pyplot as plt
import h5py
from joblib import Parallel, delayed

class BoilingDataset(Dataset):
    def __init__(self, directory):
        super().__init__()
        filenames = sorted(glob.glob(directory + '/*'))
        self._filenames = [f for f in filenames if 'plt_cnt' in f]
        with h5py.File(self._filenames[0]) as f:
            print(f.keys())
        if len(self._filenames) > 0:
            self._data = self._load_data()
            self._load_dims()

    def to_hdf5(self, filename):
        if len(self._filenames) == 0:
            return
        print(filename)
        with h5py.File(filename, 'w') as f:
            f.create_dataset('temperature', data=self._data['temp'])
            f.create_dataset('velx', data=self._data['velx'])
            f.create_dataset('vely', data=self._data['vely'])
            f.create_dataset('dfun', data=self._data['dfun'])
            f.create_dataset('pressure', data=self._data['pres'])
            f.create_dataset('x', data=self._data['x'])
            f.create_dataset('y', data=self._data['y'])
            #f.create_dataset('real-runtime-params', data=f['real runtime parameters'][:])
            #f.create_dataset('int-runtime-params', data=f['integer runtime parameters'][:])

    def _load_data(self):
        frame_dicts = self._load_files()
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

    def _runtime_params(self, f, key):
        with h5py.File(self._filenames[0], 'r') as f:
            return f[key][:]

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
                var_dict[key] = np.empty((nyb, nxb))
                for block in blocks:
                    a, b, _ = block.get_relative_loc([frame.xmin, frame.ymin, 0])
                    r, c = b * y_bs, a * x_bs 
                    var_dict[key][r:r+y_bs, c:c+x_bs] = block[key]

            var_dict['x'] = np.empty((nyb, nxb))
            var_dict['y'] = np.empty((nyb, nxb))
            block_idx = 0
            for block in blocks:
                x, y = np.meshgrid(block.xrange('center'),
                                   block.yrange('center'))
                a, b, _ = block.get_relative_loc([frame.xmin, frame.ymin, 0])
                r, c = b * y_bs, a * x_bs 
                var_dict['x'][r:r+y_bs,c:c+x_bs] = x 
                var_dict['y'][r:r+y_bs,c:c+x_bs] = y

            frames.append(var_dict)
        return frames

def unblock_dataset(write_dir, read_dir):
    b = BoilingDataset(read_dir)
    dir_name = read_dir[read_dir.find('gravX-'):]
    b.to_hdf5(f'{target}/{dir_name}.hdf5')

if __name__ == '__main__':
    target = str(Path.home() / 'crsp/ai4ts/share/FlowBoilingStudies/Gravity-FC72-2D_HDF5/')
    base = str(Path.home() / 'crsp/ai4ts/share/FlowBoilingStudies/Gravity-FC72-2D/')
    subdirs = [f for f in glob.glob(f'{base}/*') if 'gravX-' in f]
    print(subdirs)
    
    output = Parallel(n_jobs=len(subdirs))(delayed(unblock_dataset)(target, subdir) for subdir in subdirs)

    print('done!')
