#!/bin/bash
#SBATCH -A desi
#SBATCH -q regular
#SBATCH -t 03:00:00
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH -J p1d
#SBATCH -o logs/p1d.%j.out
#SBATCH -e logs/p1d.%j.err
#SBATCH --array=0-20   # number of variations minus 1

echo "Job started at: $(date)"

mkdir -p logs

# Define variations

# Define variations
variations=(
    # "None"
    # "nyx"
    "zmin"
    "zmax"
    "DESIY1_QMLE"
    "DESIY1_FFT3_dir"
    "data_syst_diag"
    "no_inflate"
    "no_emu_cov"
    "emu_diag"
    "emu_block"
    "cosmo"
    # "cosmo_h74"
    # "cosmo_mnu"
    "cosmo_low"
    "cosmo_high"
    "more_igm"
    "HCD0"
    "DLAs"
    "HCD_BOSS"
    "metal_thin"
    "metal_deco"
    "metal_si2"
    "metal_trad"
    "Metals_Ma2025"
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