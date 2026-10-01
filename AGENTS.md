# AGENTS.md — cup1d

## Purpose and active layout

cup1d performs cosmological inference from Lyman-alpha P1D. LaCE owns cosmology, simulation archives, and GP emulation; ForestFlow owns the optional P3D-to-P1D emulator. Avoid duplicating those implementations here.

- Public entry points: `cup1d.Analysis`, `cup1d.Args`; YAML examples in `configs/cm2026/` and `notebooks/tutorials/`.
- Configuration: `cup1d/configuration/`; data readers/mocks: `cup1d/p1ds/`; emulators: `cup1d/emulator/`; cosmology and forward model: `cup1d/theory/`.
- IGM, contaminants, and resolution: `cup1d/models/`; likelihood and named parameters: `cup1d/likelihood/`; minimization/sampling: `cup1d/inference/`; diagnostics: `cup1d/postprocessing/`.
- Deprecated paths include `cup1d/old_code/`, `cup1d/p1ds/old/`, `notebooks/old/`, and `notebooks/wip/`.

## Working rules

- Inspect `git status`, the current branch, and applicable nested instructions before editing. The maintained development branch is `vectorize`; do not switch branches or discard user changes automatically.
- Read the relevant implementation, tests, and `docs/workflow*` before changing a public interface. Follow active imports rather than assuming every notebook defines supported behavior.
- Make focused changes. Preserve scientific defaults, parameter ordering, serialization, and scalar/batch behavior unless the task explicitly changes them. Document intentional numerical changes and their validation.
- Do not edit `old_code/`, notebook `old/`, `wip/`, or developer experiments by default. Historical/paper modules are not necessarily deprecated: check callers first. Never copy an obsolete API back into the active package without checking it.
- Do not regenerate simulation archives, model weights, covariance products, chains, or publication outputs as part of a routine code change. Use configured external assets and report missing prerequisites. Never replace a scientific regression reference solely to make a test pass.
- Use Python >=3.12 and editable installs. Install sibling IGMHub repositories explicitly from compatible revisions, rather than relying on the PyPI name `lace`. Record the three commit SHAs for cross-package validation.
- Distinguish fast unit checks from model/data-dependent regression tests. A skipped or unavailable regression is not a pass. Prefer a small test of the failing scientific invariant over a test that merely reproduces the implementation.
- Maintain Jupytext `.py` notebook sources; sync only the affected pair with `jupytext --sync path/to/notebook.py`. Avoid generating every notebook for an unrelated change.
- Versions are derived from Git via setuptools-scm; do not hand-edit generated `_version.py` files. Update API docstrings and relevant documentation when behavior changes.
- Report what changed, commands actually run, missing assets/dependencies, and any numerical or scientific limitations.

## Shared scientific contracts

- Consult each package's `conventions.py`. Canonical public names include `k_iMpc`, `k_ikms`, `P1D_Mpc`, `P1D_kms`, `P3D_Mpc`, and `dkms_diMpc`. Existing serialized data and APIs retain legacy spellings; translate at explicit boundaries rather than silently renaming stored products.
- With `M = H(z)/(1+z)` in km/s/Mpc: `k_iMpc = M * k_ikms`, `P1D_kms = M * P1D_Mpc`, and P1D covariance gains two factors of M. P3D has volume units (Mpc^3); do not apply a P1D Jacobian to it.
- Preserve the distinction between comoving Mpc and Mpc/h, thermal broadening length `sigT_Mpc`, and inverse pressure smoothing scale `kF_Mpc`.
- Linear-power defaults distinguish baryon+CDM (`bc`) from total matter (`bcnu`). Check species, pivot, redshift, primordial running convention, and growth convention before comparing predictions.
- Primordial rescaling is only valid when all transfer-function/background parameters are unchanged. Changes in neutrino mass, effective relativistic species, dark energy, curvature, or densities require an appropriate fresh cosmology calculation.
- Treat scalar, redshift, k, batch, and stochastic-sample axes explicitly. Use unequal axis lengths in tests to expose accidental broadcasting; preserve ragged observational k grids.
- Covariance must preserve data ordering and selected cross-bin correlations. Validate symmetry, finite entries, and positive definiteness; do not hide invalid matrices with absolute determinants or arbitrary regularization.

## cup1d-specific safeguards

- Trace YAML defaults and overrides through `Args` and `Analysis`. Preserve sparse-variation layering and make configuration errors explicit.
- Keep physical named parameter values distinct from sampler coordinates and parameter-definition dictionaries. Preserve parameter and blob ordering through the fitter and likelihood.
- Rebinning factor one must be identity. For larger factors, respect the measurement bin edges and estimator weights; verify convergence rather than only scalar/batch agreement.
- Selecting redshifts must select the corresponding full covariance submatrix when cross-redshift covariance exists. Per-redshift diagnostics need not sum to the full correlated chi-squared.
- Verify identical finite posteriors between scalar/list and columnar batch paths, for inverse and Cholesky methods, with/without log determinants, invalid inputs, multiple data groups, and ragged grids.
- Preserve blinding in analysis products and plots. Do not expose unblinded values or change seeds/default blinding as an incidental refactor.
- Distinguish frozen data-scaled emulator covariance from parameter-dependent covariance. Dropping its log determinant is harmless for inference only when the covariance is fixed.

## Validation commands

After explicitly installing compatible LaCE and (when needed) ForestFlow checkouts:

```bash
python -m pip install -e ".[test]"
pytest -q -m "not external_model"
pytest -q -m external_model  # requires the configured external LaCE model bundle
```

`tests/test_dr1_tutorial.py` contains the baseline likelihood regression and interface checks. Run the external regression for forward-model/inference changes when assets are available. For documentation changes install `.[docs]` and run `make docs`. A full developer run is `make test`; inspect skips and asset requirements.
