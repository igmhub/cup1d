# DR1 baseline with the current code: what to run

Paper run for comparison: `~/Proyectos/projects/lya/data/out_DESI_DR1/DESIY1_QMLE3/global_opt/CH24_mpgcen_gpr/chain_7`.
Use the `lace` environment (`conda activate lace`).

## 1. Minimize (laptop, from the cup1d folder)

```bash
OUT=~/Proyectos/projects/lya/data/out_DESI_DR1
python scripts/data/data_sampler.py None       --stage minimize --path_out $OUT   # baseline (HCD in log)
python scripts/data/data_sampler.py HCD_linear --stage minimize --path_out $OUT   # HCD amplitudes linear
```

The last line printed gives the chi2 and the new folder (`.../global_opt/CH24_mpgcen_gpr/chain_N`
and `.../HCD_linear/CH24_mpgcen_gpr/chain_N`).

## 2. Look at the best fits

```bash
BASE=$OUT/DESIY1_QMLE3/global_opt/CH24_mpgcen_gpr/chain_N        # set N
LIN=$OUT/DESIY1_QMLE3/HCD_linear/CH24_mpgcen_gpr/chain_N
PAPER=$OUT/DESIY1_QMLE3/global_opt/CH24_mpgcen_gpr/chain_7

# Delta2_star-n_star vs simulations with Gauss-Newton errors, best-fit P1D, summary
python scripts/data/minimizer_plots.py $BASE $LIN --paper $PAPER --unblind
# numbers: chi2 and star parameters vs the paper (--all: every parameter)
python scripts/data/compare_minimizers.py $PAPER $BASE $LIN --unblind
```

Figures and `minimizer_summary.txt` are written in `$BASE`.

## 3. emcee with cup1d (NERSC)

Copy the baseline minimizer folder and start the sampler from it (it writes into the same folder):

```bash
rsync -av $BASE perlmutter:'$SCRATCH'/out_DESI_DR1/DESIY1_QMLE3/global_opt/CH24_mpgcen_gpr/   # laptop
# on Perlmutter, from the cup1d checkout (after git pull):
sbatch scripts/data/launch_sampler.sh $SCRATCH/out_DESI_DR1/DESIY1_QMLE3/global_opt/CH24_mpgcen_gpr/chain_N
```

## 4. Same YAML with Cobaya through lya_interface (NERSC)

The interface YAML `lya_interface/examples/dr1_cm2026_base_evaluate.yaml` is generated from
`configs/cm2026/cm2026_base.yaml` (regenerate after changing it:
`python scripts/config_from_cup1d.py ../cup1d/configs/cm2026/cm2026_base.yaml examples/dr1_cm2026_base_evaluate.yaml`).
On Perlmutter, from the lya_interface checkout:

```bash
export CONFIG=$PWD/examples/dr1_cm2026_base_evaluate.yaml
export RUN_DIR=$CFS/data/DESIY1_QMLE3/global_opt/CH24_mpgcen_gpr/cobaya_mcmc
sbatch --export=ALL,CONFIG=$CONFIG,RUN_DIR=$RUN_DIR scripts/nersc/prepare_job.sh       # best fit, covmat, YAML
sbatch --export=ALL,RUN_DIR=$RUN_DIR,NEW_CHAIN=1 scripts/nersc/mcmc_job_test.sh       # 30 min check
sbatch --export=ALL,RUN_DIR=$RUN_DIR scripts/nersc/mcmc_job.sh                        # production
```

Follow and plot it as in `lya_interface/scripts/nersc/HOWTO.md` (`chain_diagnostics.py $RUN_DIR`,
`chain_plots.py $RUN_DIR --unblind`).

## Before running on NERSC

Commit and push cup1d (`high_res`) and lya_interface, then `git pull` both on Perlmutter.
