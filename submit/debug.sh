#!/bin/bash
#SBATCH -p free-gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100:1
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

#srun torchrun \
#    --nnodes $NNODES \
#    --nproc_per_node 1 \
#    --max_restarts 0 \
#    --rdzv_backend c10d \
#    --rdzv_id $SLURM_JOB_ID \
#    --rdzv_endpoint $HOST_NODE_ADDR \
#    --redirects 3 \
#    --tee 3 \
	#data_base_dir=/share/crsp/lab/amowli/share/BubbleML2/ \
	#model_checkpoint=/pub/afeeney/train_log_dir/23755705/subcooled/Unet_vel_dataset_25_1692516617.pt
python src/train.py \
	data_base_dir=/share/crsp/lab/amowli/share/BubbleML2/ \
	dataset=PB_SubCooled \
	log_dir=/pub/afeeney/train_log_dir/ \
	experiment.distributed=False \
	experiment=temp_fno \
	experiment.train.max_epochs=3 \
	#model_checkpoint=/pub/afeeney/train_log_dir/23370470/pb_gravity/FNO_temp_input_dataset_250_1692173966.pt
