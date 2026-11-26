#!/bin/bash
# emcee from a saved minimizer result (see scripts/data/HOWTO_baseline.md).
#
# The configuration is the one recorded in minimizer_results.npy; the chain
# starts at that best fit and is written to the same chain_N folder.
#
#   sbatch scripts/data/launch_sampler.sh $SCRATCH/out_DESI_DR1/DESIY1_QMLE3/global_opt/CH24_mpgcen_gpr/chain_N
#
#SBATCH -A desi
#SBATCH -q regular
#SBATCH -t 06:00:00
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH -J p1d_sam
#SBATCH -o logs/p1d_sam.%j.out
#SBATCH -e logs/p1d_sam.%j.err

echo "Job started at: $(date)"

mkdir -p logs

results=$1
if [ -z "$results" ]; then
    echo "usage: sbatch launch_sampler.sh <chain_N folder or minimizer_results.npy>"
    exit 1
fi

source $HOME/miniconda3/bin/activate lace

cup1d_dir=$(python -c "from cup1d.utils.utils import get_path_repo; print(get_path_repo('cup1d'))")
echo "Sampling from $results"
echo "cup1d $(git -C $cup1d_dir rev-parse --short HEAD) ($(git -C $cup1d_dir branch --show-current))"

# --kill-on-bad-exit and "python -m mpi4py" make an uncaught exception on any
# rank abort the whole job instead of hanging until the walltime.
time srun -n 128 --unbuffered --kill-on-bad-exit=1 python -m mpi4py \
    $cup1d_dir/scripts/data/data_sampler.py --stage sample --results $results

echo "Job finished at: $(date)"
