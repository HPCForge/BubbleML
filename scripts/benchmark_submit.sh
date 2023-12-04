#!/bin/bash
#SBATCH -p free-gpu
#SBATCH -N 1
#SBATCH --mem-per-cpu=6G    # requesting max memory per CPU
#SBATCH --ntasks=30         # requesting 30 CPUS
#SBATCH --gres=gpu:A100:1
#SBATCH --time=016:00:00

module load python/3.10.2
module load cuda/11.7.1
module load gcc/11.2.0

source /pub/nsankar1/envs/sciml/bin/activate

python FNO_super_res_benchmark.py
