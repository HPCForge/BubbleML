# BubbleML

[![Paper](https://img.shields.io/badge/arXiv-2209.15616-blue)](https://arxiv.org/abs/2307.14623)

A multiphysics, multiphase dataset of boiling processes. These simulations can be used to model datacenter cooling systems. For instance, liquid cooling flowing across a GPU. or even cooling nuclear reactors: a pool of liquid sitting on a heated surface.

![SubCooled Temperature](video/subcooled.gif)

## Documentation and Examples

Documentation discussing the data fields, format, and relevant parameters can be found in [bubbleml_data/DOCS.md](bubbleml_data/DOCS.md). We also provide a set of [examples](examples/) illustrating how to use the dataset.

[examples/](examples/) contains Jupyter Notebooks showing how to [read and visualize BubbleML](examples/data_loading.ipynb)
and [train a Fourier Neural Operator](examples/pytorch_training.ipynb) on the BubbleML dataset. These are stand-alone examples that use a small, downsampled version of
Subcooled Pool boiling. These examples are intended to show 1. how to load the dataset, 2. how to read tensors from
the dataset, and 3. how to setup model training for the dataset. Extended descriptions can be found in [bubbleml_data/DOCS.md](bubbleml_data/DOCS.md). To run the examples, you should follow the [environment setup](sciml/README.md) for the SciML code.

## Download BubbleML

BubbleML is publicly available and open source. We provide links to download each study in [bubbleml_data/README.md](bubbleml_data/README.md).

## Extending BubbleML

It's possible that BubbleML will not match your needs. For instance in BubbleML's current iteration, each study varies one parameter. One obvious extension is to vary multiple parameters, like the heater temperature and liquid temperature. This will lead to different phenomena. Another idea is runnning low resolution simulations to study upscaling models. And, of course, there are some labs who may just want to generate very large datasets, containing hundreds or thousands of individual simulations!

To support such efforts, we provide a [reproducibility capsule](https://github.com/Lab-Notebooks/Outflow-Forcing-BubbleML) for running your own boiling simulations with Flash-X. This includes lab notebooks for running simulations. It also includes analysis scripts and the submissions files used to generate BubbleML. The 2D simulations are fairly quick.

## Models

Checkpoints for the models mentioned in the paper, along with ther respective results are listed in the [model zoo](model-zoo/README.md). (Note: metrics will not necissarily match the paper. We hope that this page serves as a "live" listing that shows the best results thus far.)

## Running SciML Code

Please refer to the [SciML README.md](sciml/README.md)

## Running Optical Flow Benchmarks

Please refer to the [Optical Flow README.md](optical_flow/README.md)

## Citation

If you find this dataset useful in your research, please consider citing the following paper:

```bibtex
@inproceedings{
    hassan2023bubbleml,
    title={Bubble{ML}: A Multi-Physics Dataset and Benchmarks for Machine Learning},
    author={Sheikh Md Shakeel Hassan and Arthur Feeney and Akash Dhruv and Jihoon Kim and 
            Youngjoon Suh and Jaiyoung Ryu and Yoonjin Won and Aparna Chandramowlishwaran},
    booktitle={Advances in Neural Information Processing Systems},
    year={2023},
    url={https://openreview.net/forum?id=0Wmglu8zak}
}
```
