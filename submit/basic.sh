#!/bin/bash
#SBATCH -A amowli_lab_gpu
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:A30:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=20:00:00

module load anaconda/2022.05
. ~/.mycondaconf
conda activate bubble-sciml
module load gcc/11.2.0

python src/train.py \
	data_base_dir=/share/crsp/lab/ai4ts/share/BubbleML/ \
	log_dir=/share/crsp/lab/ai4ts/afeeney/log_dir \
	dataset=PB_SubCooled \
	experiment=temp_unet2d \
	model_checkpoint=/data/homezvol2/afeeney/crsp/ai4ts/afeeney/thermal_models/subcooled/UNet2d_temp_input_dataset_500_1690005305.pt \
