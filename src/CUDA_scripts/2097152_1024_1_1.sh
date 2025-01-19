#!/bin/bash
#SBATCH --partition=edu5
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --job-name=2097152_1024_1_1
#SBATCH --output=2097152_1024_1_1.out
#SBATCH --error=2097152_1024_1_1.err
srun marzola.out 2097152 1 1024 1
            