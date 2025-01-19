#!/bin/bash
#SBATCH --partition=edu5
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --job-name=4194304_1024_4_1
#SBATCH --output=4194304_1024_4_1.out
#SBATCH --error=4194304_1024_4_1.err
srun marzola.out 4194304 4 1024 1
            