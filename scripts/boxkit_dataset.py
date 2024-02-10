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
        self._filenames = [f for f in filenames if 'plt_cnt' in f][:-1]
        with h5py.File(self._filenames[0]) as f:
            print(f.keys())
        if len(self._filenames) > 0:
            self._data = self._load_data()
            self._load_dims()

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
            
            REAL_RUNTIME_PARAMS = 'real runtime parameters'
            INT_RUNTIME_PARAMS = 'integer runtime parameters'

            if REAL_RUNTIME_PARAMS in f.keys():
                f.create_dataset('real-runtime-params', data=f[REAL_RUNTIME_PARAMS][:])
            if INT_RUNTIME_PARAMS in f.keys():
                f.create_dataset('int-runtime-params', data=f[INT_RUNTIME_PARAMS][:])

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

    def _runtime_params(self, f, key):
        with h5py.File(self._filenames[0], 'r') as f:
            return f[key][:]

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
            var_dict[key] = np.empty((nyb, nxb))
            for block in blocks:
                r = y_bs * round(int((nyb * (block.ymin - frame.ymin))/(frame.ymax - frame.ymin))/y_bs)
                c = x_bs * round(int((nxb * (block.xmin - frame.xmin))/(frame.xmax - frame.xmin))/x_bs)
                var_dict[key][r:r+y_bs, c:c+x_bs] = block[key]

        var_dict['x'] = np.empty((nyb, nxb))
        var_dict['y'] = np.empty((nyb, nxb))
        block_idx = 0
        for block in blocks:
            x, y = np.meshgrid(block.xrange('center'),
                               block.yrange('center'))
            r = y_bs * round(int((nyb * (block.ymin - frame.ymin))/(frame.ymax - frame.ymin))/y_bs)
            c = x_bs * round(int((nxb * (block.xmin - frame.xmin))/(frame.xmax - frame.xmin))/x_bs)
            var_dict['x'][r:r+y_bs,c:c+x_bs] = x 
            var_dict['y'][r:r+y_bs,c:c+x_bs] = y

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
    target = str(Path.home() / '/share/crsp/lab/ai4ts/share/simul_ts_0.1/SubCooled-FC72-2D_HDF5/')
    Path(target).mkdir(parents=True, exist_ok=True)

    base = str(Path.home() / '/share/crsp/lab/ai4ts/share/simul_ts_0.1/SubCooled-FC72-2D/')

    subdirs = [f for f in glob.glob(f'{base}/*') if TWALL in f]
    print(subdirs)
    
    for idx, subdir in enumerate(subdirs):
        print(f'processing {subdir} {idx}/{len(subdirs)}')
        unblock_dataset(target, subdir)
    
    print('done!')
