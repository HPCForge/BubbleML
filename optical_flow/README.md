## Helper files for optical flow of boiling datasets

This directory provides helper files for running optical flow experiments with the official [RAFT](https://github.com/princeton-vl/RAFT) and [GMFlow](https://github.com/haofeixu/gmflow).

### Creation of BubbleML optical flow dataset
An optical flow dataset can be created from an uncompressed folder of a BubbleML study using the script
```console
python create_opticalflow_dataset.py --ip_dir /path/to/BubbleML/study/ --op_dir /path/to/optical-flow-datasets/Boiling/
```

The dataloaders provided for RAFT and GMFlow are slight modifications of the original implementations in the respective repositories to enable the training of models using BubbleML data. Copy the respective files to `core/datasets.py` in case of RAFT and `data/datasets.py`.

The jupyter notebooks can be copied to the `RAFT` and `GMFlow` home directories to observe performance of optical flow models on the BubbleML data.
