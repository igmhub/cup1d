"""Compare best fits: chi2, star parameters and every parameter.

Accepts chain_N folders (or result files) holding either the current
``minimizer_results.npy`` / ``sampler_results.npy`` or the legacy
``fitter_results.npy`` of the DR1 paper runs. The star parameters of current
results need the theory, so the emulator is loaded (a minute or two).

    python scripts/data/compare_minimizers.py \
        <paper chain_7> <new baseline chain_N> <new HCD_linear chain_N>

The first folder is the reference for the differences. Parameters within
2% of a flat prior edge (unit-cube coordinate < 0.02 or > 0.98) are marked
with ``<`` or ``>``. Values are in each run's own parameterization (column
``otype``): log and linear HCD amplitudes are not directly comparable.
"""

import argparse
import os
from pathlib import Path

import numpy as np

STAR = ("Delta2_star", "n_star", "alpha_star")
EDGE = 0.02


def result_file(path):
    path = Path(path)
    if path.is_file():
        return path
    for name in ("minimizer_results.npy", "sampler_results.npy", "fitter_results.npy"):
        if (path / name).is_file():
            return path / name
    raise FileNotFoundError(f"no result file in {path}")


def load(path, unblind=False):
    """name -> {chi2, star, values, cube, otypes} of one best fit."""
    path = result_file(path)
    payload = np.load(path, allow_pickle=True).item()
    if payload.get("format_version") == 1:
        from cup1d.inference.analysis import Analysis

        analysis = Analysis.from_results(path)
        fitter = analysis.fitter
        names = payload["parameter_names"]
        if "parameter_bounds" in payload:
            bounds = payload["parameter_bounds"]
        else:  # results written before the bounds were stored
            print(f"{path}: no stored prior bounds, using the current ones", flush=True)
            bounds = {n: (p["min_value"], p["max_value"]) for n, p in fitter.like.free_params.items()}
        cube = np.asarray(payload["fit"]["mle_cube"], dtype=float)
        chi2 = float(payload["fit"]["mle_chi2"])
        star = {name: float(fitter.mle_cosmo[name]) for name in STAR}  # blinded
        if unblind:
            star = fitter.like.blind.unblind(star)
        otypes = payload.get("parameter_otypes", {})
    else:  # legacy DR1 paper result
        from cup1d.postprocessing.chains.products import load_parameter_bounds

        fit = payload["fitter"]
        names, bounds = load_parameter_bounds(str(path.parent))
        cube = np.asarray(fit["mle_cube"], dtype=float)
        chi2 = -2 * float(fit["lnprob_mle"])
        star = {name: float(fit["mle_cosmo"][name]) for name in STAR}  # blinded
        if unblind:  # offsets of the DR1 paper data
            from cup1d.postprocessing.literature import load_blinding_dr1
            from cup1d.utils.blinding import Blinding

            star = Blinding(load_blinding_dr1()).unblind(star)
        otypes = {}
    values = {n: bounds[n][0] + (bounds[n][1] - bounds[n][0]) * c for n, c in zip(names, cube)}
    return dict(label=str(path.parent), chi2=chi2, star=star, values=values,
                cube=dict(zip(names, cube)), otypes=otypes)


def edge_mark(fit, name):
    c = fit["cube"].get(name)
    if c is None:
        return " "
    return "<" if c < EDGE else (">" if c > 1 - EDGE else " ")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", nargs="+", help="chain_N folders or result files; the first is the reference")
    parser.add_argument("--all", action="store_true", help="print every parameter, not only the star ones")
    parser.add_argument("--unblind", action="store_true",
                        help="remove the blinding offsets of the star parameters (DESI DR1 is unblinded); "
                        "by default they are compared blinded, with the same offsets for every run")
    args = parser.parse_args()

    fits = [load(r, unblind=args.unblind) for r in args.results]
    print("star parameters", "unblinded" if args.unblind else "blinded (same offsets for every run)")
    ref = fits[0]
    width = 16
    for i, fit in enumerate(fits):
        print(f"[{i}] {fit['label']}")

    def cell(value, reference, digits):
        if reference is None:
            return f"{value:>{width}.{digits}f}"
        return f"{value:.{digits}f} ({value - reference:+.{digits}f})".rjust(width + 4)

    header = f"{'':22s}" + "".join(f"[{i}]".rjust(width if i == 0 else width + 4) for i in range(len(fits)))
    print("\n" + header + "   (difference to [0] in brackets)")
    print(f"{'chi2':22s}" + "".join(
        cell(f["chi2"], None if f is ref else ref["chi2"], 2) for f in fits))
    for name in STAR:
        print(f"{name:22s}" + "".join(
            cell(f["star"][name], None if f is ref else ref["star"][name], 4) for f in fits))

    names = list(dict.fromkeys(n for f in fits for n in f["values"]))
    if args.all:
        print(f"\n{'parameter':22s}" + "".join(f"{f'[{i}]':>{width}s}" for i in range(len(fits))) + "  otype")
        for name in names:
            cells = []
            for f in fits:
                v = f["values"].get(name)
                cells.append(f"{'-':>{width}s}" if v is None else f"{v:>{width - 1}.4g}{edge_mark(f, name)}")
            otype = ",".join(sorted({str(f["otypes"].get(name, "")) for f in fits} - {""}))
            print(f"{name:22s}" + "".join(cells) + f"  {otype}")

    print("\nparameters within 2% of a prior edge:")
    for i, f in enumerate(fits):
        edges = [f"{n}{edge_mark(f, n)}" for n in f["cube"] if edge_mark(f, n) != " "]
        print(f"[{i}] " + (", ".join(edges) if edges else "none"))


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
