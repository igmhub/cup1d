# DR1 baseline with the current code: what to run

Comparisons:

* **cup1d vs Cobaya (lya_interface)**, same configuration, with either emulator:
  LaCE GP (`configs/cm2026/cm2026_base.yaml`, the CM2026 baseline; `None` below) or
  ForestFlow GP (`configs/cm2026/cm2026_base_forestmpggp.yaml`). Minimizations:
  `lya_interface/notebooks/07_dr1_p1d_lace_cobaya_vs_cup1d` (laptop).
* **LaCE baseline vs DR1 paper**, cup1d.

Paper run: `$OUT/DESIY1_QMLE3/global_opt/CH24_mpgcen_gpr/chain_7`. Use the `lace` environment.
New runs go to `<data>/<variation or global_opt>/<emulator>/chain_N`, with emulator
`lace_mpg` or `forest_mpg_gp`.

## 1. Minimize (laptop, from the cup1d folder)

```bash
OUT=~/Proyectos/projects/lya/data/out_DESI_DR1
FF=configs/cm2026/cm2026_base_forestmpggp.yaml
python scripts/data/data_sampler.py $FF        --stage minimize --path_out $OUT   # ForestFlow GP
python scripts/data/data_sampler.py None       --stage minimize --path_out $OUT   # LaCE baseline (HCD in log)
python scripts/data/data_sampler.py HCD_linear --stage minimize --path_out $OUT   # LaCE, HCD linear
```

The last lines printed give the chi2, the folder, the parameters near a prior
bound, and whether the best fit is inside the emulator training hull. The
minimizer also saves the Gauss-Newton errors (the walkers of step 3 start from
them) and the frozen configuration: `--stage sample` uses that, not the YAML.

## 2. Look at the best fits

```bash
FFMIN=$OUT/DESIY1_QMLE3/global_opt/forest_mpg_gp/chain_N     # set N
BASE=$OUT/DESIY1_QMLE3/global_opt/lace_mpg/chain_N
LIN=$OUT/DESIY1_QMLE3/HCD_linear/lace_mpg/chain_N
PAPER=$OUT/DESIY1_QMLE3/global_opt/CH24_mpgcen_gpr/chain_7

# Delta2_star-n_star vs simulations with Gauss-Newton errors, best-fit P1D, summary (written in the first folder)
python scripts/data/minimizer_plots.py $BASE $LIN $FFMIN --paper $PAPER --unblind
# numbers: chi2 and star parameters vs the paper (--all: every parameter)
python scripts/data/compare_minimizers.py $PAPER $BASE $LIN $FFMIN --unblind
```

## 3. emcee with cup1d (NERSC)

Copy a minimizer folder and start the sampler from it (it writes into the same folder):

```bash
rsync -av $FFMIN perlmutter:'$SCRATCH'/out_DESI_DR1/DESIY1_QMLE3/global_opt/forest_mpg_gp/   # laptop
# on Perlmutter, from the cup1d checkout (after git pull):
sbatch scripts/data/launch_sampler.sh $SCRATCH/out_DESI_DR1/DESIY1_QMLE3/global_opt/forest_mpg_gp/chain_N
```

Same for the LaCE baseline (`.../global_opt/lace_mpg/chain_N`).

emcee runs one ensemble of `n_walkers` x ndim walkers whose evaluations are
shared by all MPI ranks (`mcmc.mpi_ensemble: independent` for the former one
ensemble per rank). The chain is written to `chain_N/emcee_chain.h5` as it runs:
if the job is killed, submit it again and it continues. At the end the log
prints the autocorrelation time (steps/tau should be > 50), the acceptance and
the fraction of points outside the training hull.

## 4. Same configuration with Cobaya through lya_interface (NERSC)

The interface YAMLs are generated from the cup1d YAMLs (regenerate after changing them):

```bash
python scripts/config_from_cup1d.py ../cup1d/configs/cm2026/cm2026_base.yaml examples/dr1_cm2026_base_evaluate.yaml
python scripts/config_from_cup1d.py ../cup1d/configs/cm2026/cm2026_base_forestmpggp.yaml examples/dr1_cm2026_forestmpggp_evaluate.yaml
```

At the same point they give the native cup1d chi2 within 1e-4 (LaCE) and 1e-3 (ForestFlow).
Each chain folder freezes its interface and native configurations, so regenerating
the YAMLs does not change a running chain. On Perlmutter, from the lya_interface checkout (LaCE shown; for
ForestFlow use `dr1_cm2026_forestmpggp_evaluate.yaml` and `.../forest_mpg_gp/cobaya_mcmc`):

```bash
CONFIG=$PWD/examples/dr1_cm2026_base_evaluate.yaml
RUN_DIR=$CFS/data/DESIY1_QMLE3/global_opt/lace_mpg/cobaya_mcmc
sbatch --export=ALL,CONFIG=$CONFIG,RUN_DIR=$RUN_DIR scripts/nersc/prepare_job.sh   # best fit, covmat, YAML (CONFIG required)
sbatch --export=ALL,RUN_DIR=$RUN_DIR,NEW_CHAIN=1 scripts/nersc/mcmc_job_test.sh    # 30 min check
sbatch --export=ALL,RUN_DIR=$RUN_DIR scripts/nersc/mcmc_job.sh                     # production
```

Follow and plot it as in `lya_interface/scripts/nersc/HOWTO.md` (`chain_diagnostics.py $RUN_DIR`,
`chain_plots.py $RUN_DIR --unblind`). The job resources were tuned for the hires ForestFlow
run: check the time per step in the 30 min test.

## Before running on NERSC

`git pull` cup1d, lya_interface, LaCE and ForestFlow on Perlmutter.
