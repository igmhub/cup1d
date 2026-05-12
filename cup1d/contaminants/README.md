# cup1d/contaminants

Contaminant modeling module for the Lyman-alpha forest.

## Description

This module provides classes for modeling metals and high column density (HCD) systems that contaminate Lyman-alpha forest measurements:

- **Base Contaminant** (`base_contaminants.py`) - Base class for all contaminants
- **HCD Models** - Various HCD contamination models
- **Si** - Silicon II and III metal line contamination
- **AGN/Supernova** - AGN and supernova feedback effects

## Classes

| Class | Description |
|-------|-------------|
| `Contaminant` | Base class for contaminant modeling |
| `HCD_BOSS` | Walther et al. (2024) HCD model (Eq. 5.2) |
| `HCD_Model_Rogers` | Rogers et al. (2018) HCD model |
| `HCD_Model_McDonald2005` | McDonald et al. (2005) HCD model |
| `AGN_Model` | Chabanier et al. (2020), AGN contamination model (Eq. 21) |
| `SN_Model` | Viel et al. (2013) Supernova feedback model |
| `SiAdd` | Chaves-Montero et al. (2026) Additive Si contamination |
| `SiMult` | Chaves-Montero et al. (2026) Multiplicative Si contamination |
| `SiVid` | Ma et al. (2026) SiIII contamination |

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

## See Also

- [cup1d.igm](../igm) - IGM modeling
- [cup1d.likelihood](../likelihood) - Likelihood framework
