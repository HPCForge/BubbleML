from op_lib.metrics import fourier
from op_lib.hdf5_dataset import TempDataset

d = TempDataset('/data/homezvol2/afeeney/crsp/ai4ts/share/PB_simulation/SubCooled-FC72-2D_HDF5/Twall-100.hdf5',
                time_window=3)
item, label = d[0]


fourier(item[0].unsqueeze(0), label, 8, 8)
