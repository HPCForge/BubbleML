# BubbleML Downloads

BubbleML is hosted on AWS and can be publicly downloaded. Each boiling study can be downloaded separately:

| Study | Size |
|-----------------------|----|
| [Single Bubble](https://bubble-ml-simulations.s3.us-east-2.amazonaws.com/single-bubble.tar.gz)     | 503.0 MB |
| [Pool Boiling Saturated](https://bubble-ml-simulations.s3.us-east-2.amazonaws.com/pool-boiling-saturated-fc72-2d.tar.gz)      | 24.4 GB |
| [Pool Boiling Subcooled](https://bubble-ml-simulations.s3.us-east-2.amazonaws.com/pool-boiling-subcooled-fc72-2d.tar.gz)      | 10.5 GB |
| [Pool Boiling Gravity](https://bubble-ml-simulations.s3.us-east-2.amazonaws.com/pool-boiling-gravity-fc72-2d.tar.gz)        | 16.5 GB |
| [Flow Boiling Inlet Velocity](https://bubble-ml-simulations.s3.us-east-2.amazonaws.com/flow-boiling-velscale-fc72-2d.tar.gz) | 11.4 GB |
| [Flow Boiling Gravity](https://bubble-ml-simulations.s3.us-east-2.amazonaws.com/flow-boiling-gravity-fc72-2d.tar.gz)        | 10.9 GB |
| [Pool Boiling Subcooled 0.1](https://bubble-ml-simulations.s3.us-east-2.amazonaws.com/pool-boiling-subcooled-fc72-2d-0.1.tar.gz) | 155.1 GB |
| [Pool Boiling Gravity 0.1](https://bubble-ml-simulations.s3.us-east-2.amazonaws.com/pool-boiling-gravity-fc72-2d-0.1.tar.gz) | 163.8 GB |
| [Flow Boiling Gravity 0.1](https://bubble-ml-simulations.s3.us-east-2.amazonaws.com/flow-boiling-gravity-fc72-2d-0.1.tar.gz) | 108.6 GB |
| [3D Pool Boiling Earth Gravity](https://bubble-ml-simulations.s3.us-east-2.amazonaws.com/pool-boiling-earth-gravity-3d.tar.gz)    | 122.2 GB |
| [3D Pool Boiling ISS Gravity](https://bubble-ml-simulations.s3.us-east-2.amazonaws.com/pool-boiling-iss-gravity-3d.tar.gz) | 62.6 GB |
| [3D Flow Boiling Earth Gravity](https://bubble-ml-simulations.s3.us-east-2.amazonaws.com/flow-boiling-earth-gravity-3d.tar.gz) | 93.9 GB |

Each download is a `.tar.gz`. They can be unzipped using the command 

```console
tar -xvf <study>.tar.gz -C /path/to/BubbleML/<study>/
```

After unzipping, you will see a collection of hdf5 files in `/path/to/BubbleML/<study>/.
Each hdf5 file corresponds to one simulation. The hdf5 files can be loaded with common libraries,
such as `h5py`. 

## Documentation and Examples

We provide [documentation](DOCS.md) describing the different hdf5 datasets in each simulation file.
There are also [examples](../examples) showing how to load a BubbleML simulation, list out it's datasets, 
visualize the different simulation fields, and access the metadata. 

## Bulk Download

The studies can also be downloaded in bulk by running the bash script 

```console
bash download_all.sh
```

This will download all datasets listed above. Note: the full dataset is over a terabyte in size.
