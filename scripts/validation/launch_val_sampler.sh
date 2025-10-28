#!/bin/bash
#SBATCH -A desi
#SBATCH -q regular
#SBATCH -t 03:00:00
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH -J p1d
#SBATCH -o logs/p1d.%j.out
#SBATCH -e logs/p1d.%j.err
#SBATCH --array=0-4   # number of variations minus 1

echo "Job started at: $(date)"

mkdir -p logs

# Define variations

variations=(
    "mpg_central"
    # "mpg_central_igm"
    # "mpg_central_igm0"
    "nyx_central"
    "sherwood"
)

source /global/homes/j/jjchaves/miniconda3/bin/activate lace

# Pick variation for this array task
var=${variations[$SLURM_ARRAY_TASK_ID]}
echo "Job $SLURM_ARRAY_JOB_ID, task $SLURM_ARRAY_TASK_ID: running variation = $var"

time srun -n 128 --unbuffered python /global/homes/j/jjchaves/cup1d/scripts/validation/val_sampler.py $var

echo "Job finished at: $(date)"