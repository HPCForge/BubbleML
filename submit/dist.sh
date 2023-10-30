#!/bin/bash
#SBATCH -A amowli_lab_gpu
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:A30:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=5G
#SBATCH --time=48:00:00

# One node needs to be used as the "host" for the rendezvuoz
# system used by torch. This just gets a list of the hostnames
# used by the job, and selects the first one.
HOST_NODE_ADDR=$(scontrol show hostnames | head -n 1)
NNODES=$(scontrol show hostnames | wc -l)

module load anaconda/2022.05
. ~/.mycondaconf
conda activate bubble-sciml

#DATASET=PB_SubCooled_0.1
#DATASET=FB_Gravity_0.1

DATASET=PB_SubCooled
#DATASET=PB_WallSuperHeat
#DATASET=PB_Gravity
#DATASET=FB_Gravity
#DATASET=FB_InletVel

#EXPERIMENT=temp_unet2d
#EXPERIMENT=temp_unet_mod_attn
#EXPERIMENT=temp_ufnet
#EXPERIMENT=temp_fno
#EXPERIMENT=temp_uno
#EXPERIMENT=temp_ffno

# GFNO requires multi-gpu...
# Do this last
#EXPERIMENT=temp_gfno

#data_base_dir=/share/crsp/lab/amowli/share/simul_ts_0.1/ \

srun torchrun \
    --nnodes $NNODES \
    --nproc_per_node 1 \
    --max_restarts 0 \
    --rdzv_backend c10d \
    --rdzv_id $SLURM_JOB_ID \
    --rdzv_endpoint $HOST_NODE_ADDR \
    --redirects 3 \
    --tee 3 \
    sciml/train.py \
		data_base_dir=/share/crsp/lab/amowli/share/BubbleML2/ \
		log_dir=/pub/afeeney/train_log_dir \
		dataset=$DATASET \
		experiment=$EXPERIMENT \
		experiment.train.max_epochs=150 \
