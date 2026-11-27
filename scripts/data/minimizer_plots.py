"""Figures of a minimizer result: star parameters vs simulations, best-fit P1D.

Writes into the chain folder of the first result (``--out`` to change it):

* ``minimizer_star_simulations``: Delta2_star--n_star of the emulator training
  simulations, the best fit with its 1 and 2 sigma Gauss--Newton ellipses
  (flat priors included as Gaussians of the same variance), and
  the best fits of the other results given.
* ``minimizer_best_fit_p1d``: model against the data.
* ``minimizer_summary.txt``: chi2, star parameters with errors, parameters
  near prior edges.

Usage::

    python scripts/data/minimizer_plots.py <chain_N> [<other chain_N> ...] \
        [--paper <DR1 paper chain_7>] [--unblind]

Results are ``minimizer_results.npy`` folders; ``--paper`` takes a legacy
``fitter_results.npy`` folder (drawn without errors). Star parameters are
blinded unless ``--unblind`` (the offsets are the same for every DR1 run).
"""

import argparse
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", nargs="+", help="chain_N folders with minimizer_results.npy; the first is the main one")
    parser.add_argument("--paper", default=None, help="DR1 paper chain folder (fitter_results.npy) to compare with")
    parser.add_argument("--unblind", action="store_true", help="remove the blinding offsets (DESI DR1)")
    parser.add_argument("--out", default=None, help="output folder (default: the first chain folder)")
    parser.add_argument("--no-p1d", action="store_true", help="skip the best-fit P1D figure")
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    from cup1d.postprocessing.minimizer import MinimizerPlotter, legacy_fit_entry

    main_fit, *others = [MinimizerPlotter.from_results(r, unblind=args.unblind) for r in args.results]
    if args.out:
        main_fit.out_folder = main_fit.out_folder.__class__(args.out)
    if args.paper:
        others.append(legacy_fit_entry(args.paper, label="DR1 paper", unblind=args.unblind))
    print(main_fit.plot_all(others=others, skip=("best_fit_p1d",) if args.no_p1d else ()))
    print(f"figures in {main_fit.out_folder}")


if __name__ == "__main__":
    main()
