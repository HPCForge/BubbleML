#!/bin/bash
#SBATCH -p free-gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=4G
#SBATCH --time=0:10:00

# One node needs to be used as the "host" for the rendezvuoz
# system used by torch. This just gets a list of the hostnames
# used by the job, and selects the first one.
HOST_NODE_ADDR=$(scontrol show hostnames | head -n 1)
NNODES=$(scontrol show hostnames | wc -l)

module load anaconda/2022.05
. ~/.mycondaconf
conda activate bubble-sciml

export TORCH_DISTRIBUTED_DEBUG=DETAIL

#srun torchrun \
#    --nnodes $NNODES \
#    --nproc_per_node 1 \
#    --max_restarts 0 \
#    --rdzv_backend c10d \
#    --rdzv_id $SLURM_JOB_ID \
#    --rdzv_endpoint $HOST_NODE_ADDR \
#    --redirects 3 \
#    --tee 3 \
python src/train.py \
	data_base_dir=/share/crsp/lab/amowli/share/BubbleML2/ \
	log_dir=/pub/afeeney/train_log_dir/ \
	dataset=PB_Gravity\
	experiment.distributed=False \
	experiment=temp_unet2d \
	experiment.train.max_epochs=3 \
	experiment.lr_scheduler.patience=50
	model_checkpoint=/pub/afeeney/train_log_dir/23370440/pb_gravity/
