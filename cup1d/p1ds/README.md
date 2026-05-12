# cup1d/p1ds

1D Power Spectrum (P1D) Data Loading Module.

## Description

This module provides classes for loading 1D power spectrum measurements from various simulations and observations:

- **Base Classes** - `BaseDataP1D`, `BaseMockP1D`
- **Observational Data** - DESI Y1, eBOSS, Chabanier2019, etc.
- **Simulation Data** - Nyx, Gadget, Illustris, etc.

## Data Sources

| Class | Source | Reference |
|-------|--------|-----------|
| `DataDESIY1` | DESI Y1 | DESI Collaboration (2024) |
| `DataChabanier2019` | Chabanier2019 | Chabanier et al. (2019) |
| `DataIrsic2017` | Irsic2017 | Irsic et al. (2017) |
| `DataWalther2018` | Walther2018 | Walther et al. (2018) |
| `DataNyx` | Nyx simulation | Armitage et al. (2018) |
| `DataGadget` | Gadget simulation | Springel et al. (2005) |

## Usage

```python
from cup1d.p1ds import DataDESIY1

# Load DESI Y1 data
data = DataDESIY1()
print(f"Redshifts: {data.z}")
print(f"Wavenumbers: {data.k_kms}")
```

## Scientific References

- [DESI Collaboration (2024)](https://ui.adsabs.harvard.edu/abs/2024arXiv240401056D) - DESI Y1 results
- [Chabanier et al. (2019)](https://ui.adsabs.harvard.edu/abs/2019MNRAS.489.5787C) - Lyman-alpha forest P1D
- [Karacayli et al. (2022)](https://ui.adsabs.harvard.edu/abs/2022MNRAS.514.4914K) - CHIME P1D constraints
- [Walther et al. (2018)](https://ui.adsabs.harvard.edu/abs/2018arXiv180900012W) - HIRES P1D constraints
- [Irsic et al. (2017)](https://ui.adsabs.harvard.edu/abs/2017PhRvL.119c1102I) - XQ-100 P1D

## See Also

- [cup1d.likelihood](../likelihood) - Likelihood framework
- [cup1d.cosmology](../cosmology) - Cosmology utilities