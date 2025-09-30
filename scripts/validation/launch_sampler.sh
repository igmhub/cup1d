#!/bin/bash
#SBATCH -A desi
#SBATCH -q regular
#SBATCH -t 03:00:00
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH -J p1d
#SBATCH -o logs/p1d.%j.out
#SBATCH -e logs/p1d.%j.err

echo "Job started at: $(date)"

mkdir -p logs

# Define variations

source /global/homes/j/jjchaves/miniconda3/bin/activate lace
time srun -n 128 --unbuffered python /global/homes/j/jjchaves/cup1d/scripts/validation/val_sampler.py

echo "Job finished at: $(date)"