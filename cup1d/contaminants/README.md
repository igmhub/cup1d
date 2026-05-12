# cup1d/contaminants

Contaminant modeling module for the Lyman-alpha forest.

## Description

This module provides classes for modeling metal-line contaminants and HCD (Hubble Canvas Dissolution) systems that contaminate Lyman-alpha forest measurements:

- **Base Contaminant** (`base_contaminants.py`) - Base class for all contaminants
- **HCD Models** - Various HCD contamination models
- **Si IV** - Silicon IV metal line contamination
- **AGN/Supernova** - AGN and supernova feedback effects

## Classes

| Class | Description |
|-------|-------------|
| `Contaminant` | Base class for contaminant modeling |
| `HCDModelRogers` | Rogers et al. (2018) HCD model |
| `HCDModelMcDonald2005` | McDonald et al. (2005) HCD model |
| `HCD_BOSS` | BOSS HCD model |
| `AGNModel` | AGN contamination model |
| `SNModel` | Supernova feedback model |

## Usage

```python
from cup1d.contaminants import Contaminant

# Create contaminant model
contam = Contaminant(
    coeffs=coeffs,
    list_coeffs=list_coeffs,
    prop_coeffs=prop_coeffs,
    free_param_names=free_param_names
)
```

## Scientific References

- [McDonald et al. (2006)](https://ui.adsabs.harvard.edu/abs/2006ApJ...653..815M) - Metal line contaminants
- [Rogers et al. (2018)](https://ui.adsabs.harvard.edu/abs/2018MNRAS.474.3841R) - HCD modeling
- [Chabanier et al. (2019)](https://ui.adsabs.harvard.edu/abs/2019MNRAS.489.5787C) - Full contamination model
- [Onorbe et al. (2019)](https://ui.adsabs.harvard.edu/abs/2019MNRAS.485.4815O) - HCD constraints

## See Also

- [cup1d.igm](../igm) - IGM modeling
- [cup1d.likelihood](../likelihood) - Likelihood framework