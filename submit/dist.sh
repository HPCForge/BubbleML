#!/bin/bash
#SBATCH -A amowli_lab_gpu
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:A30:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=04:00:00

# One node needs to be used as the "host" for the rendezvuoz
# system used by torch. This just gets a list of the hostnames
# used by the job, and selects the first one.
HOST_NODE_ADDR=$(scontrol show hostnames | head -n 1)
NNODES=$(scontrol show hostnames | wc -l)

module load anaconda/2022.05
. ~/.mycondaconf
conda activate bubble-sciml

srun torchrun \
    --nnodes $NNODES \
    --nproc_per_node 2 \
    --max_restarts 0 \
    --rdzv_backend c10d \
    --rdzv_id $SLURM_JOB_ID \
    --rdzv_endpoint $HOST_NODE_ADDR \
    --redirects 3 \
    --tee 3 \
    src/train.py \
	data_base_dir=/share/crsp/lab/ai4ts/share/BubbleML/ \
	log_dir=/share/crsp/lab/ai4ts/afeeney/log_dir \
	dataset=PB_SubCooled \
	experiment=temp_unet2d \
	experiment.train.max_epochs=2 \
	experiment.train.batch_size=2 \
	#experiment.lr_scheduler.patience=50
	#model_checkpoint=/share/crsp/lab/ai4ts/afeeney/log_dir/23089030/subcooled/UNet2d_vel_dataset_100_1691046606.pt \
