import torch
from torch.utils.data import ConcatDataset, Dataset
import h5py
import random
from torchvision.transforms import Resize
import torchvision.transforms.functional as TF
from pathlib import Path

# The early timesteps of a simulation may be "unsteady"
# We say that the simulation enters a steady state around
# timestep 30.
STEADY_TIME = 30

class HDF5ConcatDataset(ConcatDataset):
    def __init__(self, datasets):
        super().__init__(datasets)

    def absmax_vel(self):
        return max(d.absmax_vel() for d in self.datasets)

    def absmax_temp(self):
        return max(d.absmax_temp() for d in self.datasets)

    def normalize_temp_(self, absmax_temp=None):
        if not absmax_temp:
            absmax_temp = self.absmax_temp()
        for d in self.datasets:
            d.normalize_temp_(absmax_temp)
        return absmax_temp

    def normalize_vel_(self, absmax_vel=None):
        if not absmax_vel:
            absmax_vel = self.absmax_vel()
        for d in self.datasets:
            d.normalize_vel_(absmax_vel)
        return absmax_vel

class HDF5Dataset(Dataset):
    def __init__(self, filename, transform=False, time_window=1, future_window=1):
        super().__init__()
        assert time_window > 0, 'HDF5Dataset.__init__():time window should be positive'
        self.transform = transform
        self.time_window = time_window
        self.future_window = future_window
        self._data = {}
        with h5py.File(filename, 'r') as f:
            self._data['temp'] = torch.nan_to_num(torch.from_numpy(f['temperature'][..., STEADY_TIME:]))
            self._data['velx'] = torch.nan_to_num(torch.from_numpy(f['velx'][..., STEADY_TIME:]))
            self._data['vely'] = torch.nan_to_num(torch.from_numpy(f['vely'][..., STEADY_TIME:]))
            self._data['dfun'] = torch.from_numpy(f['dfun'][..., STEADY_TIME:])
            self._data['x'] = torch.from_numpy(f['x'][..., STEADY_TIME:])
            self._data['y'] = torch.from_numpy(f['y'][..., STEADY_TIME:])

        self._redim_temp(filename)


    def _redim_temp(self, filename):
        r"""
        Each hdf5 file non-dimensionalizes temperature to the same range. 
        If the wall temperature is varied across simulations, then the temperature
        must be re-dimensionalized, so it can be properly normalized across
        simulations.
        this is ONLY DONE WHEN THE FILENAME INCLUDES Twall-
        """
        filename = Path(filename).stem
        wall_temp = None
        TWALL = 'Twall-'
        if TWALL in filename:
            self._data['temp'] *= int(filename[len(TWALL):])
            print('wall temp', self._data['temp'].max())

    def absmax_temp(self):
        return self._data['temp'].abs().max()

    def absmax_vel(self):
        return max(self._data['velx'].abs().max(), self._data['vely'].abs().max())

    def normalize_temp_(self, scale):
        assert scale >= self._data['temp'].max(), 'temp scale = {scale} too small'
        self._data['temp'] = 2 * (self._data['temp'] / scale) - 1

    def normalize_vel_(self, scale):
        for v in ('velx', 'vely'):
            assert scale >= self._data[v].max(), f'{v} scale = {scale} too small'
            self._data[v] = self._data[v] / scale

    def get_x(self):
        return self._data['x'][..., self.time_window:]
    
    def get_dy(self):
        r""" dy is the grid spacing in the y direction.
        """
        return self._data['y'][0, 0, 0]

    def get_dfun(self):
        return self._data['dfun'][..., self.time_window:]

    def __len__(self):
        # len is the number of timesteps. Each prediction
        # requires time_window frames, so we can't predict for
        # the first few frames.
        # we may also predict several frames in the future, so we
        # can't include those in length
        return self._data['temp'].size(2) - self.time_window - (self.future_window - 1) 

    def _transform(self, input, label):
        if self.transform:
            if random.random() > 0.5:
                input = TF.hflip(input)
                label = TF.hflip(label)
        return input, label

    def __getitem__(self, timestep):
        assert False, 'Not Implemented'

class TempInputDataset(HDF5Dataset):
    r""" 
    This is a dataset for predicting only temperature. It assumes that
    velocities are known in every timestep. It also enables writing
    past predictions for temperature and using them to make future
    predictions.
    """
    def __init__(self, filename, transform=False, time_window=1, future_window=1):
        super().__init__(filename, transform, time_window, future_window)
        self.in_channels = 3 * self.time_window
        self.out_channels = self.future_window

    def _get_input_stack(self, timestep):
        return torch.stack([
            self._data['temp'][..., timestep],
            self._data['velx'][..., timestep],
            self._data['vely'][..., timestep],
        ], dim=0)

    def _get_label(self, timestep):
        return self._data['temp'][..., timestep]

    def __getitem__(self, timestep):
        input = torch.cat([self._get_input_stack(timestep + k) for k in range(self.time_window)], dim=0)
        #input = torch.cat([
        #    input,
        #    self._data['velx'][..., timestep + self.time_window].unsqueeze(0),
        #    self._data['vely'][..., timestep + self.time_window].unsqueeze(0),
        #], dim=0)
        label = torch.stack([self._get_label(timestep + k) for k in range(self.future_window)], dim=0)
        return self._transform(input, label)

    def write_temp(self, temp, timestep):
        if temp.dim() == 2:
            temp.unsqueeze_(-1)
        base_time = timestep + self.time_window
        self._data['temp'][..., base_time:base_time + self.future_window] = temp

class TempVelDataset(HDF5Dataset):
    r"""
    This is a dataset for predicting both temperature and velocity.
    Velocities and temperatures are unknown. The model writes past
    predictions to reuse for future predictions.
    """
    def __init__(self, filename, transform=False, time_window=1):
        super().__init__(filename, transform, time_window)
        self.in_channels = 4 * self.time_window
        self.out_channels = 3

    def _get_stack(self, timestep):
        cur_velx = self._data['velx'][..., timestep]
        cur_vely = self._data['velx'][..., timestep]
        #cur_dfun = self._data['dfun'][..., timestep]
        cur_dfun = self._data['dfun'][..., timestep]
        cur_dfun[cur_dfun < 0] = 0
        # zero out the liquid velocities
        #cur_velx[cur_dfun < 0] = 0
        #cur_vely[cur_dfun < 0] = 0
        return torch.stack([
            self._data['temp'][..., timestep],
            cur_velx,
            cur_vely,
            cur_dfun
        ], dim=0)
    
    def __getitem__(self, timestep):
        input = torch.cat([self._get_stack(timestep + k) for k in range(self.time_window)], dim=0)
        print(input.size(), self.time_window)

        #cur_velx = self._data['velx'][..., timestep + self.time_window]
        #cur_vely = self._data['velx'][..., timestep + self.time_window]
        #cur_dfun = self._data['dfun'][..., timestep + self.time_window]
        # zero out the liquid velocities
        #cur_velx[cur_dfun < 0] = 0
        #cur_vely[cur_dfun < 0] = 0

        #input = torch.cat([
        #    input,
        #    cur_velx.unsqueeze(0),
        #    cur_vely.unsqueeze(0)
        #])

        label = torch.stack([
            self._data['temp'][..., timestep + self.time_window],
            self._data['velx'][..., timestep + self.time_window], 
            self._data['vely'][..., timestep + self.time_window]
        ], dim=0)

        return self._transform(input, label)

    def write_velx(self, velx, timestep):
        self._data['velx'][..., timestep + self.time_window] = velx

    def write_vely(self, vely, timestep):
        self._data['vely'][..., timestep + self.time_window] = vely

    def write_temp(self, temp, timestep):
        self._data['temp'][..., timestep + self.time_window] = temp
