# BubbleML

A multi-physics dataset of boiling processes.  
This repository includes downloads, visualizations, and sample applications.

![SubCooled Temperature](video/subcooled.gif)

This dataset can be used to train operator networks, act as a ground truth for Physics-Informed Neural Networks, or train computer vision models.
These models have applications to cooling systems for datacenters (I.e., liquid cooling flowing across a GPU) or even cooling nuclear reactors (I.e., a pool of liquid sitting on a heated surface).

This repository provides baselines and sample applications for the bubble ML dataset. Sample videos can be seen in [video](video/) directory. 

## Dataset Downloads

The dataset is hosted on AWS. Each boiling study can be downloaded separately.

| Study | Size |
|-----------------------|----|
| [Single Bubble](https://bubble-ml-simulations.s3.us-east-2.amazonaws.com/single-bubble.tar.gz)     | 206.0 MB |
| [Pool Boiling Saturated](https://bubble-ml-simulations.s3.us-east-2.amazonaws.com/pool-boiling-saturated-fc72-2d.tar.gz)      | 24.4 GB |
| [Pool Boiling Subcooled](https://bubble-ml-simulations.s3.us-east-2.amazonaws.com/pool-boiling-subcooled-fc72-2d.tar.gz)      | 10.5 GB |
| [Pool Boiling Gravity](https://bubble-ml-simulations.s3.us-east-2.amazonaws.com/pool-boiling-gravity-fc72-2d.tar.gz)        | 16.5 GB |
| [Flow Boiling Inlet Velocity](https://bubble-ml-simulations.s3.us-east-2.amazonaws.com/flow-boiling-velscale-fc72-2d.tar.gz) | 11.4 GB |
| [Flow Boiling Gravity](https://bubble-ml-simulations.s3.us-east-2.amazonaws.com/flow-boiling-gravity-fc72-2d.tar.gz)        | 10.9 GB |
| [3D Pool Boiling Earth Gravity](https://anl.app.box.com/s/wwj2f9b0t2eetjmieoj163axmxctuswd)    | 140.7 GB |
| [3D Pool Boiling Low Gravity](https://anl.app.box.com/s/vnsfq59k9gnkhxyhhrc48sjwj61sjnia/) | 71.5 GB |

## Environment Setup
The code assumes access to a fairly modern Nvidia GPU, though
it may also work on AMD GPUs if PyTorch is installed with Rocm support.
Results have been reproduced on a Linux cluster with V100, A30, and A100 GPUs using PyTorch 2.0 and CUDA 11.7.

To install dependencies, we recommend creating a conda environment:

~~~~
conda create -n bubble-sciml -f conda/bubbleml-pytorch-2.0.1-cuda-11.7.yaml
~~~~

## Running Sample Code

The sample code uses Hydra to manage different configurations.
For example, we treat each simulation group as a dataset: `conf/dataset/*.yaml`.
Similarly, each model is treated as a separate experiment: `conf/experiment/*.yaml`.
Running the main training script with different settings with run an experiment on a particular dataset.

For example, training a temperature prediction UNet model on the subcooled boiling dataset is simple:

~~~~
python src/train.py dataset=PB_SubCooled experiment=temp_unet2d
~~~~

If you want to run a pretrained model, you can specify the `model_checkpoint` path

~~~~
python src/train.py dataset=PB_SubCooled experiment=temp_unet2d model_checkpoint=<path>
~~~~
