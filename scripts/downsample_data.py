r"""
Downsample a dataset so it can be kept in the repo to use as an example
"""


import h5py
import numpy as np

src_dir = '/share/crsp/lab/amowli/share/BubbleML2/PoolBoiling-SubCooled-FC72-2D/'
dst_dir = '/share/crsp/lab/amowli/share/BubbleML2/example/'
files = ['Twall-100.hdf5', 'Twall-103.hdf5', 'Twall-106.hdf5']


src_files = [src_dir + f for f in files]
dst_files = [dst_dir + f for f in files]

keys_to_downsample = [
    'temperature',
    'velx',
    'vely',
    'dfun',
    'pressure',
    'x',
    'y'
]

keys_to_copy = [
    'real-runtime-params',
    'int-runtime-params'
]

for src_file, dst_file in zip(src_files, dst_files):
    with h5py.File(src_file, 'r') as sf:
        with h5py.File(dst_file, 'w') as df:
            for key in keys_to_downsample:
                df.create_dataset(key, data=sf[key][:, ::8,::8])
            for key in keys_to_copy:
                df.create_dataset(key, data=sf[key])
