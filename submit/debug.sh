#!/bin/bash
#SBATCH -p free-gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:A30:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=4G
#SBATCH --time=0:30:00

# One node needs to be used as the "host" for the rendezvuoz
# system used by torch. This just gets a list of the hostnames
# used by the job, and selects the first one.
HOST_NODE_ADDR=$(scontrol show hostnames | head -n 1)
NNODES=$(scontrol show hostnames | wc -l)

module load anaconda/2022.05
. ~/.mycondaconf
conda activate bubble-sciml

export TORCH_DISTRIBUTED_DEBUG=DETAIL

python sciml/train.py \
	data_base_dir=/share/crsp/lab/amowli/share/BubbleML2/ \
	dataset=PB_Gravity \
	log_dir=/pub/afeeney/train_log_dir/ \
	experiment.distributed=False \
	experiment=unet_arena/pb_temp \
	experiment.train.max_epochs=1 \
	model_checkpoint=/pub/afeeney/final_model_checkpoints/PB_Gravity/unet_mod.pt
