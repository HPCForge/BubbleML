#!/bin/bash
#SBATCH -A amowli_lab_gpu
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=5G
#SBATCH --time=24:00:00

# One node needs to be used as the "host" for the rendezvuoz
# system used by torch. This just gets a list of the hostnames
# used by the job, and selects the first one.
HOST_NODE_ADDR=$(scontrol show hostnames | head -n 1)
NNODES=$(scontrol show hostnames | wc -l)

module load anaconda/2022.05
. ~/.mycondaconf
conda activate bubble-sciml

python scripts/fno_overfit.py
