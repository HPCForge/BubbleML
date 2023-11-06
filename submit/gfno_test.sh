#!/bin/bash
#SBATCH -A amowli_lab_gpu
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=3G
#SBATCH --time=24:00:00

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

#DATASET=PB_SubCooled
#DATASET=PB_WallSuperHeat
DATASET=PB_Gravity
#DATASET=FB_Gravity
#DATASET=FB_InletVel

#MODEL=fno
#MODEL=uno
#MODEL=ffno
#MODEL=gfno
#MODEL=unet_bench
#MODEL=unet_arena
#MODEL=ufnet

#srun torchrun \
#    --nnodes $NNODES \
#    --nproc_per_node 1 \
#    --max_restarts 0 \
#    --rdzv_backend c10d \
#    --rdzv_id $SLURM_JOB_ID \
#    --rdzv_endpoint $HOST_NODE_ADDR \
#    --redirects 3 \
#    --tee 3 \
python sciml/train.py \
		data_base_dir=/share/crsp/lab/amowli/share/BubbleML2/ \
		log_dir=/pub/afeeney/train_log_dir \
		dataset=$DATASET \
		experiment=paper/gfno/tune \
		#experiment.optimizer.initial_lr=5e-4 \
		#model_checkpoint=/pub/afeeney/final_model_checkpoints/PB_Saturated/gfno.pt
		#experiment=gfno/pb_temp_large \
		#experiment.optimizer.initial_lr=5e-4
