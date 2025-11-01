#!/bin/bash
#SBATCH -A desi
#SBATCH -q regular
#SBATCH -t 03:00:00
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH -J p1d
#SBATCH -o logs/p1d.%j.out
#SBATCH -e logs/p1d.%j.err
#SBATCH --array=0-1   # number of variations minus 1

echo "Job started at: $(date)"

mkdir -p logs

# Define variations

# Define variations
variations=(
    # "None"
    # "nyx"
    # "DESIY1_QMLE"
    # "DESIY1_FFT3_dir"
    # "zmin"
    # "zmax"
    # "no_inflate"
    # "no_emu_cov"
    "emu_diag"
    "emu_block"
    # "cosmo"
    # "cosmo_low"
    # "cosmo_high"
    # "data_syst_diag"
    # "metal_trad"
    # "metal_si2"
    # "metal_thin"
    # "metal_deco"
    # "more_igm"
    # "DLAs"
    # "HCD0"
    # "Metals_Ma2025"
    # "HCD_BOSS"
    #############
    # "DESIY1_FFT_dir"
    # "less_igm"
    # "no_res"
    # "metals_z"
    # "hcd_z"
    # "Turner24"
    # "kF_kms"
)

source /global/homes/j/jjchaves/miniconda3/bin/activate lace

# Pick variation for this array task
var=${variations[$SLURM_ARRAY_TASK_ID]}
echo "Job $SLURM_ARRAY_JOB_ID, task $SLURM_ARRAY_TASK_ID: running variation = $var"

time srun -n 128 --unbuffered python /global/homes/j/jjchaves/cup1d/scripts/data/data_sampler.py $var

echo "Job finished at: $(date)"