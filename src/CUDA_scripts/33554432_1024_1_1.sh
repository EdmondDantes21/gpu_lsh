#!/bin/bash
#SBATCH --partition=edu5
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --job-name=33554432_1024_1_1
#SBATCH --output=33554432_1024_1_1.out
#SBATCH --error=33554432_1024_1_1.err
srun marzola.out 33554432 1 1024 1
            