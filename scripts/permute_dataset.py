r"""
Convert an HDF5 file laid out [XxYxT] to [TxXxY].
"""

import argparse
import glob
import h5py
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--src', type=str, help='path to hdf5 file to permute')
parser.add_argument('--dst', type=str, help='path to write permuted file')
args = parser.parse_args()

keys_to_permute = [
    'temperature',
    'velx',
    'vely',
    'pressure',
    'dfun',
    'x',
    'y'
]
    
keys_to_copy = [
    'real-runtime-params',
    'int-runtime-params'
]

# change from [XxYxT] to [TxXxY]
perm = (2, 0, 1)

src_files = [Path(fn) for fn in glob.glob(f'{args.src}/*.hdf5')]
print(src_files)

for src_file in src_files:
    with h5py.File(src_file, 'r') as src:
        dst_file = f'{args.dst}/{src_file.name}'
        print(f'copying {src_file} to {dst_file}')
        with h5py.File(dst_file, 'w') as dst:
            for key in keys_to_permute:
                dst.create_dataset(key, data=src[key][:].transpose(perm))
            for key in keys_to_copy:
                dst.create_dataset(key, data=src[key][:])
