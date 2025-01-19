#!/bin/bash
#SBATCH --partition=edu5
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --job-name=33554432_1024_8_0
#SBATCH --output=33554432_1024_8_0.out
#SBATCH --error=33554432_1024_8_0.err
srun marzola.out 33554432 8 1024 0
            