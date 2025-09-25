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
#SBATCH --array=0-11   # number of variations minus 1

echo "Job started at: $(date)"

mkdir -p logs

# Define variations
variations=(
    # "no_inflate"
    # "no_emu_cov"
    "no_inflate_no_emu_cov"
    "cosmo"
    "metal_trad"
    "metal_si2"
    "metal_deco"
    "metal_thin"
    "no_res"
    "Turner24"
    "more_igm"
    "less_igm"
    "metals_z"
    "hcd_z"
)

source /global/homes/j/jjchaves/miniconda3/bin/activate lace

# Pick variation for this array task
var=${variations[$SLURM_ARRAY_TASK_ID]}
echo "Job $SLURM_ARRAY_JOB_ID, task $SLURM_ARRAY_TASK_ID: running variation = $var"

time srun --unbuffered python /global/homes/j/jjchaves/cup1d/src/cup1d/scripts/data/profile_like.py $var

echo "Job finished at: $(date)"