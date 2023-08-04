#!/bin/bash
#SBATCH -p standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=4:00:00

module load anaconda/2022.05
. ~/.mycondaconf
conda activate bubble-sciml 
module load gcc/11.2.0

python scripts/boxkit_dataset.py
