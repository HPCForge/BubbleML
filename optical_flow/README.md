## Helper files for optical flow of boiling datasets

This directory provides helper files for running optical flow experiments with the official [RAFT](https://github.com/princeton-vl/RAFT) and [GMFlow](https://github.com/haofeixu/gmflow).

### Creation of BubbleML optical flow dataset
An optical flow dataset can be created from an uncompressed folder of a BubbleML study using the script
```console
python create_opticalflow_dataset.py --ip_dir /path/to/BubbleML/study/ --op_dir /path/to/optical-flow-datasets/Boiling/
```

The dataloaders provided for RAFT and GMFlow are slight modifications of the original implementations in the respective repositories to enable the training of models using BubbleML data. Copy the respective files to `core/datasets.py` in case of RAFT and `data/datasets.py` in case of GMFlow.
The finetuning process can then be performed using the scripts given below: 

#### GMFlow
```console
python main.py --checkpoint_dir chairs_boil/ --resume pretrained/gmflow_chairs-1d776046.pth --stage boiling --batch_size 8 --num_workers 4  --lr 1e-6 --weight_decay 1e-6 --image_size 512 512    --save_ckpt_freq 1000 --num_steps 2000
python main.py --checkpoint_dir things_boil/ --resume pretrained/gmflow_things-e9887eda.pth --stage boiling --batch_size 8 --num_workers 4  --lr 1e-6 --weight_decay 1e-6 --image_size 512 512    --save_ckpt_freq 1000 --num_steps 2000
python main.py --checkpoint_dir sintel_boil/ --resume pretrained/gmflow_sintel-0c07dcb3.pth --stage boiling --batch_size 8 --num_workers 4  --lr 1e-6 --weight_decay 1e-6 --image_size 512 512    --save_ckpt_freq 1000 --num_steps 2000
```
#### RAFT
```console
python -u train.py --name raft-boiling-chairs --stage boiling  --restore_ckpt /data/homezvol1/sheikhh1/RAFT/models/raft-chairs.pth --gpus 0 --num_steps 2000 --batch_size 5 --lr 0.000001 --image_size 512 512 --wdecay 0.000001 --mixed_precision
python -u train.py --name raft-boiling-things --stage boiling  --restore_ckpt /data/homezvol1/sheikhh1/RAFT/models/raft-things.pth --gpus 0 --num_steps 2000 --batch_size 5 --lr 0.000001 --image_size 512 512 --wdecay 0.000001 --mixed_precision
python -u train.py --name raft-boiling-sintel --stage boiling  --restore_ckpt /data/homezvol1/sheikhh1/RAFT/models/raft-sintel.pth --gpus 0 --num_steps 2000 --batch_size 5 --lr 0.000001 --image_size 512 512 --wdecay 0.000001 --mixed_precision
```

The jupyter notebooks can be copied to the `RAFT` and `GMFlow` home directories to observe performance of optical flow models on the BubbleML data.
