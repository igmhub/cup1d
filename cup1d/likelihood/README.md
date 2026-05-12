# cup1d/likelihood

Likelihood Framework for Lyman-alpha Forest Analysis.

## Description

This module provides the core likelihood framework for Bayesian inference of cosmological parameters from Lyman-alpha forest P1D measurements:

- **Core Likelihood** (`likelihood.py`) - Main likelihood class
- **MCMC Fitting** (`fitter.py`) - Monte Carlo sampling
- **Theory Models** (`lya_theory.py`, `model_igm.py`, etc.) - Physical models
- **Minimization** (`iminuit_minimizer.py`) - Parameter optimization

## Key Classes

| Class | Description |
|-------|-------------|
| `Likelihood` | Core likelihood class for P1D analysis |
| `Fitter` | MCMC sampler using Cobaya |
| `LyaTheory` | Theory predictions for Lyman-alpha forest |
| `ModelIGM` | IGM physical model |
| `ModelContaminants` | Contaminant model |

## Usage

```python
from cup1d.likelihood import Likelihood

# Create likelihood
like = Likelihood(
    data=data,
    theory=theory,
    free_param_names=["Delta2_star", "n_star"],
    free_param_limits=[(0.5, 2.5), (0.8, 1.2)]
)

# Compute log-likelihood
log_like = like.get_log_like(values)
```

## Scientific References

- [Chabanier et al. (2019)](https://ui.adsabs.harvard.edu/abs/2019MNRAS.489.5787C) - Lyman-alpha forest constraints
- [DESI Collaboration (2024)](https://ui.adsabs.harvard.edu/abs/2024arXiv240401056D) - DESI Y1 BAO
- [Planck Collaboration (2020)](https://ui.adsabs.harvard.edu/abs/2020A&A...641...6P) - Planck 2018 results

## See Also

- [cup1d.p1ds](../p1ds) - P1D data loading
- [cup1d.igm](../igm) - IGM modeling
- [cup1d.contaminants](../contaminants) - Contaminant modeling