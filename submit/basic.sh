#!/bin/bash
#SBATCH -A amowli_lab_gpu
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=016:00:00

module load anaconda/2022.05
. ~/.mycondaconf
conda activate bubble-sciml
module load gcc/11.2.0

python src/train.py \
	data_base_dir=/share/crsp/lab/ai4ts/share/simul_ts_0.1/ \
	log_dir=/share/crsp/lab/ai4ts/afeeney/log_dir \
	dataset=PB_SubCooled_0.1 \
	experiment=temp_unet2d \
	experiment.train.max_epochs=2 \
	#experiment.lr_scheduler.patience=50
	#model_checkpoint=/share/crsp/lab/ai4ts/afeeney/log_dir/23089030/subcooled/UNet2d_vel_dataset_100_1691046606.pt \
	#model_checkpoint=/data/homezvol2/afeeney/crsp/ai4ts/afeeney/thermal_models/subcooled/UNet2d_temp_input_dataset_500_1690005305.pt \
