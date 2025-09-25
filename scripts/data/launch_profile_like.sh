#!/bin/bash
#SBATCH -A desi
#SBATCH -q regular
#SBATCH -t 01:00:00
#SBATCH -N 1
#SBATCH -n 128
#SBATCH -C cpu
#SBATCH -J p1d
#SBATCH -o logs/p1d.%j.out
#SBATCH -e logs/p1d.%j.err

echo "Job started at: $(date)"

mkdir -p logs

source /global/homes/j/jjchaves/miniconda3/bin/activate lace

time srun --unbuffered python /global/homes/j/jjchaves/cup1d/src/cup1d/scripts/data/profile_like.py

echo "Job finished at: $(date)"