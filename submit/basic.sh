#!/bin/bash
#SBATCH -p free-gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:A30:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=4:00:00

module load anaconda/2022.05
. ~/.mycondaconf
conda activate mf-pytorch2
module load gcc/11.2.0

python src/train.py \
	dataset=PB_SubCooled \
	experiment=temp_uno \
	experiment.torch_dataset_name=temp_input_dataset
