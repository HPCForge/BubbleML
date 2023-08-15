#!/bin/bash
#SBATCH -p free
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --time=4:00:00

module load anaconda/2022.05
. ~/.mycondaconf
conda activate bubble-sciml 
module load gcc/11.2.0


DATASET=FlowBoiling-VelScale-FC72-2D
DATASET=PoolBoiling-Gravity-FC72-2D
SRC=/share/crsp/lab/ai4ts/share/BubbleML/$DATASET
DST=/share/crsp/lab/amowli/share/BubbleML2/$DATASET

mkdir -p $DST

#python scripts/boxkit_dataset.py
python scripts/permute_dataset.py --src $SRC --dst $DST 
