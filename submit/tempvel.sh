#!/bin/bash
#SBATCH -p free-gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:A30:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=016:00:00

export MASTER_ADDR=$(hostname)
export MASTER_PORT=12345
export WORLD_SIZE=$(($SLURM_NTASKS))
export RANK=0
python sciml/train.py \
	data_base_dir=/share/crsp/lab/ai4ts/share/simul_ts_0.1 \
	log_dir=/pub/junchc2/BubbleML \
	dataset=PB_SubCooled_0.1 \
	experiment=paper/unet_arena/pb_temp