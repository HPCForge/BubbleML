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

class HDF5ConcatDataset(ConcatDataset):
    def __init__(self, datasets):
        super().__init__(datasets)

    def future_window(self):
        return self.datasets[0].future_window

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

    def datum_dim(self):
        return self.datasets[0].datum_dim()

class HDF5Dataset(Dataset):
    def __init__(self,
                 filename,
                 steady_time,
                 transform=False,
                 time_window=1,
                 future_window=1,
                 push_forward_steps=1):
        super().__init__()
        assert time_window > 0, 'HDF5Dataset.__init__():time window should be positive'
        self.filename = filename
        self.steady_time = steady_time
        self.transform = transform
        self.time_window = time_window
        self.future_window = future_window
        self.push_forward_steps = push_forward_steps
        self.temp_scale = None
        self.vel_scale = None
        self.reset()

    def reset(self):
        self._data = {}
        with h5py.File(self.filename, 'r') as f:
            self._data['temp'] = torch.nan_to_num(torch.from_numpy(f['temperature'][:][self.steady_time:]))
            self._data['velx'] = torch.nan_to_num(torch.from_numpy(f['velx'][:][self.steady_time:]))
            self._data['vely'] = torch.nan_to_num(torch.from_numpy(f['vely'][:][self.steady_time:]))
            self._data['dfun'] = torch.nan_to_num(torch.from_numpy(f['dfun'][:][self.steady_time:]))
            self._data['x'] = torch.from_numpy(f['x'][:][self.steady_time:])
            self._data['y'] = torch.from_numpy(f['y'][:][self.steady_time:])

        self._redim_temp(self.filename)
        if self.temp_scale and self.vel_scale:
            self.normalize_temp_(self.temp_scale)
            self.normalize_vel_(self.vel_scale)

    def datum_dim(self):
        return self._data['temp'].size()

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
        self._data['temp'] = 2 * (self._data['temp'] / scale) - 1
        self.temp_scale = scale

    def normalize_vel_(self, scale):
        for v in ('velx', 'vely'):
            self._data[v] = self._data[v] / scale
        self.vel_scale = scale

    def get_x(self):
        return self._data['x'][self.time_window:]
    
    def get_dy(self):
        r""" dy is the grid spacing in the y direction.
        """
        return self._data['y'][0, 0, 0]

    def get_dfun(self):
        return self._data['dfun'][self.time_window:]

    def _get_temp(self, timestep):
        return self._data['temp'][timestep]

    def _get_vel_stack(self, timestep):
        return torch.stack([
            self._data['velx'][timestep],
            self._data['vely'][timestep],
        ], dim=0)

    def _get_coords(self, timestep):
        x = self._data['x'][timestep]
        x /= x.max()
        y = self._data['y'][timestep]
        y /= y.max()
        coords = torch.stack([
            x, y
        ], dim=0)
        return coords

    def _get_dfun(self, timestep):
        vapor_mask = self._data['dfun'][timestep] > 0
        return vapor_mask.to(float) - 0.5

    def __len__(self):
        # len is the number of timesteps. Each prediction
        # requires time_window frames, so we can't predict for
        # the first few frames.
        # we may also predict several frames in the future, so we
        # can't include those in length
        return self._data['temp'].size(0) - self.time_window - (self.future_window * self.push_forward_steps - 1) 

    def _transform(self, *args):
        if self.transform:
            if random.random() > 0.5:
                args = tuple([TF.hflip(arg) for arg in args])
        return args

    def __getitem__(self, timestep):
        assert False, 'Not Implemented'

class TempInputDataset(HDF5Dataset):
    r""" 
    This is a dataset for predicting only temperature. It assumes that
    velocities are known in every timestep. It also enables writing
    past predictions for temperature and using them to make future
    predictions.
    """
    def __init__(self,
                 filename,
                 steady_time,
                 use_coords,
                 transform=False,
                 time_window=1,
                 future_window=1,
                 push_forward_steps=1):
        super().__init__(filename, steady_time, transform, time_window, future_window, push_forward_steps)
        coords_dim = 2 if use_coords else 0
        self.in_channels = 3 * self.time_window + coords_dim + 2 * self.future_window
        self.out_channels = self.future_window

    def __getitem__(self, timestep):
        coords = self._get_coords(timestep)
        temps = torch.stack([self._get_temp(timestep + k) for k in range(self.time_window)], dim=0)
        vel = torch.cat([self._get_vel_stack(timestep + k) for k in range(self.time_window + self.future_window)], dim=0) 
        base_time = timestep + self.time_window 
        label = torch.stack([self._get_temp(base_time + k) for k in range(self.future_window)], dim=0)
        return (coords, *self._transform(temps, vel, label))

    def write_temp(self, temp, timestep):
        if temp.dim() == 2:
            temp.unsqueeze_(-1)
        base_time = timestep + self.time_window
        self._data['temp'][base_time:base_time + self.future_window] = temp

class TempVelDataset(HDF5Dataset):
    r"""
    This is a dataset for predicting both temperature and velocity.
    Velocities and temperatures are unknown. The model writes past
    predictions to reuse for future predictions.
    """
    def __init__(self,
                 filename,
                 steady_time,
                 use_coords,
                 transform=False,
                 time_window=1,
                 future_window=1,
                 push_forward_steps=1):
        super().__init__(filename, steady_time, transform, time_window, future_window, push_forward_steps)
        coords_dim = 2 if use_coords else 0
        self.temp_channels = self.time_window
        self.vel_channels = self.time_window * 2
        self.dfun_channels = self.time_window

        self.in_channels = coords_dim + self.temp_channels + self.vel_channels + self.dfun_channels
        self.out_channels = 3 * self.future_window

    def _get_timestep(self, timestep):
        r"""
        Get the window rooted at timestep.
        This includes the {timestep - self.time_window, ..., timestep - 1} as input
        and {timestep, ..., timestep + future_window - 1} as output
        """
        coords = self._get_coords(timestep)
        temp = torch.stack([self._get_temp(timestep + k) for k in range(self.time_window)], dim=0)
        vel = torch.cat([self._get_vel_stack(timestep + k) for k in range(self.time_window)], dim=0) 
        dfun = torch.stack([self._get_dfun(timestep + k) for k in range(self.time_window)], dim=0)

        base_time = timestep + self.time_window 
        temp_label = torch.stack([self._get_temp(base_time + k) for k in range(self.future_window)], dim=0)
        vel_label = torch.cat([self._get_vel_stack(base_time + k) for k in range(self.future_window)], dim=0)
        return self._transform(coords, temp, vel, dfun, temp_label, vel_label)

    def __getitem__(self, timestep):
        r"""
        Get the windows rooted at {timestep, timestep + self.future_window, ...}
        For each variable, the windows are concatenated into one tensor.
        """
        args = list(zip(*[self._get_timestep(timestep + k * self.future_window) for k in range(self.push_forward_steps)]))
        return tuple([torch.stack(arg, dim=0) for arg in args])

    def write_vel(self, vel, timestep):
        base_time = timestep + self.time_window
        self._data['velx'][base_time:base_time + self.future_window] = vel[0::2]
        self._data['vely'][base_time:base_time + self.future_window] = vel[1::2]

    def write_temp(self, temp, timestep):
        if temp.dim() == 2:
            temp.unsqueeze_(-1)
        base_time = timestep + self.time_window
        self._data['temp'][base_time:base_time + self.future_window] = temp
        
class VelInputDataset(HDF5Dataset):
    r""" 
    This is a dataset for predicting only velocity. It assumes that
    dfun are known at t, vel at t is also known. It also enables writing
    past predictions for velocities and using them to make future
    predictions.
    """
    def __init__(self,
                 filename,
                 steady_time,
                 use_coords,
                 transform=False,
                 time_window=1,
                 future_window=1,
                 push_forward_steps=1):
        super().__init__(filename, steady_time, transform, time_window, future_window, push_forward_steps)
        self.in_channels = 3 * self.time_window  #2 for current velocity 1 for current dfun 
        self.out_channels =2 * self.future_window #for two future velocity vx and vy 

    def __getitem__(self, timestep):
        # past velocity
        vel = torch.cat([self._get_vel_stack(timestep + k) for k in range(self.time_window)], dim=0)
        base_time = timestep + self.time_window 
        label = torch.cat([self._get_vel_stack(base_time + k) for k in range(self.future_window)], dim=0)
        # past and future dfun
        dfun = torch.stack([self._get_dfun(timestep + k) for k in range(self.time_window)], dim=0)
        vel = vel.unsqueeze(0);
        label = label.unsqueeze(0);
        dfun = dfun.unsqueeze(0);
        return self._transform(vel, dfun, label)

    def write_vel(self, vel, timestep):
        base_time = timestep + self.time_window
        self._data['velx'][base_time:base_time + self.future_window] = vel[0::2]
        self._data['vely'][base_time:base_time + self.future_window] = vel[1::2]

class VelCoordInputDataset(HDF5Dataset):
    r""" 
    This is a dataset for predicting only velocity. It assumes that
    dfun are known at t and t+1, vel at t is also known. It also enables writing
    past predictions for velocities and using them to make future
    predictions.
    """
    def __init__(self,
                 filename,
                 steady_time,
                 use_coords,
                 transform=False,
                 time_window=1,
                 future_window=1,
                 push_forward_steps=1):
        super().__init__(filename, steady_time, transform, time_window, future_window, push_forward_steps)
        coords_dim = 2 if use_coords else 0
        self.in_channels = coords_dim + 3 * self.time_window #2 for current velocity 1 for current dfun 
        self.out_channels =2 * self.future_window #for two future velocity vx and vy 

    def __getitem__(self, timestep):
        coords = self._get_coords(timestep).unsqueeze(0);
        # past velocity
        vel = torch.cat([self._get_vel_stack(timestep + k) for k in range(self.time_window)], dim=0).unsqueeze(0);
        base_time = timestep + self.time_window 
        label = torch.cat([self._get_vel_stack(base_time + k) for k in range(self.future_window)], dim=0).unsqueeze(0);
        # past and future dfun
        dfun = torch.stack([self._get_dfun(timestep + k) for k in range(self.time_window)], dim=0).unsqueeze(0);
        return self._transform(coords, vel, dfun, label)
    
    def write_vel(self, vel, timestep):
        base_time = timestep + self.time_window
        self._data['velx'][base_time:base_time + self.future_window] = vel[0::2]
        self._data['vely'][base_time:base_time + self.future_window] = vel[1::2]
