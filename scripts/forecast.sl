#!/bin/bash -l

#SBATCH -C haswell
#SBATCH --partition=debug
#SBATCH --account=desi
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --job-name=cup1d_sampler
#SBATCH --output=/global/cfs/cdirs/desi/users/font/p1d_forecast/logs/cup1d_sampler-%j.out
#SBATCH --error=/global/cfs/cdirs/desi/users/font/p1d_forecast/logs/cup1d_sampler-%j.err

# load modules to use LaCE
module load python
module load gsl
source activate cup1d

export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=1

work_dir="/global/cfs/cdirs/desi/users/font/p1d_forecast/"
echo "working dir", $work_dir

python -u $work_dir/cup1d_nersc/forecast.py --timeout 0.4 --data_cov_label Chabanier2019 

