# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: lace
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Covariance likelihood methods
#
# This comparison keeps the data, emulator, and physical initial parameter
# point fixed.  The baseline evaluates quadratic forms with stored inverse
# covariance matrices; the second YAML evaluates the identical covariance with
# Cholesky triangular solves and obtains the log determinant from its diagonal.
# The latter is numerically more stable and avoids using the inverse in the
# likelihood calculation.

# %%
from pathlib import Path

from cup1d import Analysis, Args
from cup1d.utils.utils import get_path_repo

# %%
config_dir = Path(get_path_repo("cup1d")) / "configs" / "cm2026"
configurations = {
    "inverse": config_dir / "cm2026_base.yaml",
    "cholesky": config_dir / "cm2026_base_cholesky.yaml",
}

# %%
results = {}
for label, yaml_file in configurations.items():
    args = Args.from_yaml(yaml_file, verbose=False)
    args.out_folder = f"/tmp/cup1d_covariance_{label}"
    analysis = Analysis(args)
    initial_point = analysis.fitter.initial_parameters()
    chi2 = analysis.fitter.get_chi2(initial_point)
    results[label] = chi2
    print(f"{label:8s}: initial chi2 = {chi2:.8f}")

# %% [markdown]
# Apart from round-off, the values must agree: both methods represent exactly
# the same covariance.  The Cholesky path is the appropriate default for a
# future analysis in which numerical conditioning matters.

# %%
print("difference (cholesky - inverse) =", results["cholesky"] - results["inverse"])
