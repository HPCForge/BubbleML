# Scientific Machine Learning

## Environment Setup

The code assumes access to a fairly modern Nvidia GPU, though
it may also work on AMD GPUs if PyTorch is installed with Rocm support.
Results have been reproduced on a Linux cluster with V100, A30, and A100 GPUs using PyTorch 2.0 and CUDA 11.7.

To install dependencies, we recommend creating a conda environment:

```console
conda env create -n bubble-sciml -f conda/pytorch-2.0.1-cuda-11.7.yaml
```


Our sample application code uses Hydra to manage different configurations.
For example, we treat each simulation type as a dataset: `conf/dataset/*.yaml`.
Similarly, each model is treated as a separate experiment: `conf/experiment/*.yaml`.

For example, training a temperature prediction UNet model on the subcooled boiling dataset is simple:

```console
python sciml/train.py dataset=PB_SubCooled experiment=temp_unet2d
```

If you want to run a pretrained model, you can specify the `model_checkpoint` path

```console
python sciml/train.py dataset=PB_SubCooled experiment=temp_unet2d model_checkpoint=<path>
```

The config file `conf/default.yaml` assumes that the datasets are extracted to the same location.
**This location should be set by the user. By default, this setting is empty**.
Setting the `data_base_dir`  can be done by explicity updating `conf/default.yaml` or
specifying the dataset base directory when running the python scripts.) 

For example, if you downloaded two datasets to 

```console
/your/path/to/BubbleML/saturated.hdf5
/your/path/to/BubbleML/subcooled.hdf5
```

then, to train a UNet model on the subcooled boiling dataset, just run

```console
python sciml/train.py \
	data_base_dir=/your/path/to/BubbleML \
	dataset=PB_SubCooled experiment=temp_unet
```
