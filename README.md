# BubbleML

A multi-physics dataset of boiling processes.

![SubCooled Temperature](video/subcooled.gif)

This dataset can be used to train operator networks, act as a ground truth for Physics-Informed Neural Networks, or train computer vision models.
These models have applications to cooling systems for datacenters (I.e., liquid cooling flowing across a GPU) or even cooling nuclear reactors (I.e., a pool of liquid sitting on a heated surface).

This repository provides baselines and sample applications for the bubble ML dataset. Sample videos can be seen in [video](video/) directory. 

## Dataset Downloads

The dataset is hosted on AWS. Each boiling study can be downloaded separately.

| Study | Size |
|-----------------------|----|
| [Single Bubble]()     | GB |
| [PB Saturated]()      | GB |
| [PB Subcooled](https://bubble-ml-simulations.s3.us-east-2.amazonaws.com/pool-boiling-subcooled-fc72-2d.tar.gz)      | 10.5 GB |
| [PB Gravity]()        | GB |
| [FB Inlet Velocity](https://bubble-ml-simulations.s3.us-east-2.amazonaws.com/flow-boiling-velscale-fc72-2d.tar.gz) | 11.4 GB |
| [FB Gravity]()        | GB |
| [3D Simulations]()    | GB |

## Code Installation and Usage

The code currently assumes access to a fairly modern Nvidia GPU, though
it may also work on AMD GPUs if PyTorch is installed with Rocm support.
Results have been reproduced on a Linux cluster with V100, A30, and A100 GPUs using PyTorch 2.0 and CUDA 11.7.

To install dependencies, we recommend creating a conda environment:

~~~~
conda create -f conda/environment.yaml
~~~~

We recommend installing [PyTorch](https://pytorch.org/get-started/locally/) and 
torch vision manually since there are more architecture specific options. 
For CUDA 11.7 support, 

~~~~
conda install pytorch torchvision pytorch-cuda=11.7 -c pytorch -c nvidia
~~~~

## Repository Overview

TODO: probably need to tidy things up more...

