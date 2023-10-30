#!/bin/bash
#SBATCH -A amowli_lab_gpu
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:A30:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --time=20:00:00

module load anaconda/2022.05
. ~/.mycondaconf
conda activate mf-pytorch2
module load gcc/11.2.0

python sciml/train.py \
	dataset=PB_WallSuperHeat_CrossVal150 \
	experiment=temp_unet2d \
	experiment.torch_dataset_name=temp_input_dataset \
	experiment.train.max_epochs=100
