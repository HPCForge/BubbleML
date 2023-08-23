# BubbleML Examples

This directory contains examples of how to use BubbleML that are much simpler
than our experiment code. We use a downsampled version of several Subcooled Pool Boiling datasets.
Each simulation uses every 8-th pixel, so it is much smaller and faster to train models. These
downsampled datasets have the exact same keys and metadata as the "true" datasets. The only difference
is the domain resolution.

We provide two jupyter notebooks:
1. `data_loading.ipynb` shows how to load simulations using h5py, what the keys are for each hdf5 file,
and how to visualize different timesteps for each variable. It also shows how to use the distance
function to get point in liquid or vapor, or the phase interface. 
2. `pytorch_training.ipynb` uses the three sample datasets to train a Fourier Neural Operator.
We do not apply the training strategies used in our experiments, this is intended to serve as an
example of how to use BubbleML, not reproduce our results. This example shows how to setup a
PyTorch dataset for each HDF5 file and how to use a `ConcatDataset` to combine them. It then
shows how to build a Fourier Neural Operator, run a training loop, and visualize the results.

## Running the examples:

The SciML conda environment includes jupyterlab as a dependency, so it can run the example notebooks.
In the project root directory, you can create the conda environment with pytorch, neuraloperator,
and jupyter. 

```console
conda env create -n bubble-sciml -f conda/pytorch-2.0.1-cuda-11.7.yaml
```
