#!/bin/bash
#SBATCH -A desi
#SBATCH -q regular
#SBATCH -t 01:30:00
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH -J p1d
#SBATCH -o logs/p1d.%j.out
#SBATCH -e logs/p1d.%j.err
#SBATCH --array=0-1   # number of variations minus 1

echo "Job started at: $(date)"

mkdir -p logs

# Define variations
variations=(
    # "nyx"
    # "DESIY1_QMLE"
    # "DESIY1_FFT3_dir"
    # "DESIY1_FFT_dir"
    # "no_inflate"
    # "no_emu_cov"
    # "no_inflate_no_emu_cov"
    # "cosmo"
    # "metal_trad"
    # "metal_si2"
    # "metal_deco"
    # "metal_thin"
    # "no_res"
    # "Turner24"
    # "more_igm"
    # "less_igm"
    # "metals_z"
    # "hcd_z"
    "zmin"
    "zmax"
)

source /global/homes/j/jjchaves/miniconda3/bin/activate lace

# Pick variation for this array task
var=${variations[$SLURM_ARRAY_TASK_ID]}
echo "Job $SLURM_ARRAY_JOB_ID, task $SLURM_ARRAY_TASK_ID: running variation = $var"

time srun -n 128 --unbuffered python /global/homes/j/jjchaves/cup1d/scripts/data/profile_like.py $var

echo "Job finished at: $(date)"