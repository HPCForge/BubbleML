import torch
from torch.utils.data import Dataset
import h5py
import random
from torchvision.transforms import Resize
import torchvision.transforms.functional as TF

class HDF5Dataset(Dataset):
    def __init__(self, filename, transform=False, time_window=1):
        super().__init__()
        assert time_window > 0, 'HDF5Dataset.__init__():time window should be positive'
        self.time_window = time_window
        self._data = {}
        with h5py.File(filename, 'r') as f:
            self._data['temp'] = torch.from_numpy(f['temp'][..., 30:])
            self._data['velx'] = torch.from_numpy(f['velx'][..., 30:])
            self._data['vely'] = torch.from_numpy(f['vely'][..., 30:])
            self._data['dfun'] = torch.from_numpy(f['dfun'][..., 30:])
        self.transform = transform

    def __len__(self):
        # len is the number of timesteps. Each prediction
        # requires time_window frames, so we can't predict for
        # last few frames.
        return self._data['temp'].size(2) - self.time_window 

    def _transform(self, input, label):
        if self.transform:
            if random.random() > 0.5:
                input = TF.hflip(input)
                label = TF.hflip(label)
        return input, label

    def __getitem__(self, timestep):
        assert False, 'Not Implemented'

class TempDataset(HDF5Dataset):
    def __init__(self, filename, transform=False, time_window=1):
        super().__init__(filename, transform, time_window)
        self.in_channels = 3 * self.time_window + 3
        self.out_channels = 1

    def _get_stack(self, timestep):
        return torch.stack([
            self._data['velx'][..., timestep],
            self._data['vely'][..., timestep],
            (self._data['dfun'][..., timestep] >= 0).float(),
        ], dim=0)

    def __getitem__(self, timestep):
        input = torch.cat([self._get_stack(timestep + k) for k in range(self.time_window + 1)], dim=0)
        label = self._data['temp'][..., timestep + self.time_window].unsqueeze(0)
        return self._transform(input, label)

    def write_temp(self, temp, timestep):
        r""" Function is a no-op. Intended to match
        interface of TempInputDataset
        """
        pass

class VelDataset(HDF5Dataset):
    def __init__(self, filename, transform=False, time_window=1):
        super().__init__(filename, transform, time_window)
        self.in_channels = 3 * self.time_window + 1
        self.out_channels = 2

    def _get_stack(self, timestep):
        return torch.stack([
            self._data['temp'][..., timestep],
            self._data['velx'][..., timestep],
            self._data['vely'][..., timestep],
            (self._data['dfun'][..., timestep] >= 0).float() / 3,
        ], dim=0)

    def __getitem__(self, timestep):
        input = torch.cat([self._get_stack(timestep + k) for k in range(self.time_window)], dim=0)
        input = torch.cat([
            input,
            self._data['temp'][..., timestep + self.time_window].unsqueeze(0),
            (self._data['dfun'][..., timestep + self.time_window] >= 0).float().unsqueeze(0) / 3
        ], dim=0)
        label = torch.stack([
            self._data['velx'][..., timestep + self.time_window],
            self._data['vely'][..., timestep + self.time_window],
        ], dim=0)
        return self._transform(input, label)

    def write_velx(self, velx, timestep):
        self._data['velx'][..., timestep + self.time_window] = velx

    def write_vely(self, vely, timestep):
        self._data['vely'][..., timestep + self.time_window] = vely

class TempInputDataset(HDF5Dataset):
    r""" Same as TempDataset, but includes temp in the input stack
    """
    def __init__(self, filename, transform=False, time_window=1):
        super().__init__(filename, transform, time_window)
        self.in_channels = 3 * self.time_window + 2
        self.out_channels = 1

    def _get_stack(self, timestep):
        return torch.stack([
            self._data['temp'][..., timestep],
            self._data['velx'][..., timestep] / 20,
            self._data['vely'][..., timestep] / 20,
        ], dim=0)

    def __getitem__(self, timestep):
        input = torch.cat([self._get_stack(timestep + k) for k in range(self.time_window)], dim=0)
        input = torch.cat([
            input,
            self._data['velx'][..., timestep + self.time_window].unsqueeze(0) / 20,
            self._data['vely'][..., timestep + self.time_window].unsqueeze(0) / 20,
        ], dim=0)
        label = self._data['temp'][..., timestep + self.time_window].unsqueeze(0)
        return self._transform(input, label)

    def write_temp(self, temp, timestep):
        r""" Used for testing, can write predicted temp in to use
        for future predictions
        """
        self._data['temp'][..., timestep + self.time_window] = temp

class TempVelDataset(HDF5Dataset):
    def __init__(self, filename, transform=False, time_window=1):
        super().__init__(filename, transform, time_window)
        self.in_channels = 4 * self.time_window + 1
        self.out_channels = 3

    def _get_stack(self, timestep):
        return torch.stack([
            self._data['temp'][..., timestep],
            self._data['velx'][..., timestep],
            self._data['vely'][..., timestep],
            (self._data['dfun'][..., timestep] >= 0).float()
        ], dim=0)
    
    def __getitem__(self, timestep):
        input = torch.cat([self._get_stack(timestep + k) for k in range(self.time_window)], dim=0)
        input = torch.cat([
            input,
            (self._data['dfun'][..., timestep + self.time_window] >= 0).float().unsqueeze(0)
        ], dim=0)

        label = torch.stack([
            self._data['temp'][..., timestep + self.time_window],
            self._data['velx'][..., timestep + self.time_window],
            self._data['vely'][..., timestep + self.time_window]
        ], dim=0)

        return self._transform(input, label)