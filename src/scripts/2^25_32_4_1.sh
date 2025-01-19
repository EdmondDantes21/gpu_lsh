#!/bin/bash
#SBATCH --partition=edu5
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --job-name=2^25_32_4_1
#SBATCH --output=test-%j.out
#SBATCH --error=test-%j.err
srun marzola.cu 33554432 4 32 1
            