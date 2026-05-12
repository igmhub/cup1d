# cup1d/igm

Intergalactic Medium (IGM) modeling module.

## Description

This module provides classes for modeling the physical properties of the intergalactic medium, including:
- **Temperature** (`thermal_class.py`) - Thermal broadening and temperature evolution
- **Mean Flux** (`mean_flux_class.py`) - Mean transmitted flux fraction
- **Pressure** (`pressure_class.py`) - IGM pressure history

## Classes

| Class | Description |
|-------|-------------|
| `IGM_model` | Base class for IGM modeling |
| `Thermal` | Thermal properties (sigT, gamma, T0) |
| `MeanFlux` | Mean flux fraction (tau_eff) |
| `Pressure` | Pressure modeling |

## Usage

```python
from cup1d.igm import Thermal, MeanFlux

# Create thermal model
thermal = Thermal(fid_igm=fid_igm, fid_vals=fid_vals)
T0 = thermal.get_T0(z=2.5)

# Create mean flux model
mean_flux = MeanFlux(fid_igm=fid_igm, fid_vals=fid_vals)
flux = mean_flux.get_mean_flux(z=2.5)
```

## Scientific References

- [Hui & Gnedin (1997)](https://ui.adsabs.harvard.edu/abs/1997MNRAS.292...27H) - IGM thermal history
- [McQuinn et al. (2009)](https://ui.adsabs.harvard.edu/abs/2009ApJ...694..842M) - IGM temperature evolution
- [Becker et al. (2013)](https://ui.adsabs.harvard.edu/abs/2013MNRAS.436.1023B) - Thermal history constraints
- [Faucher-Giguère et al. (2008)](https://ui.adsabs.harvard.edu/abs/2008MNRAS.387..295F) - IGM mean flux

## See Also

- [cup1d.likelihood](../likelihood) - Likelihood framework
- [cup1d.contaminants](../contaminants) - Contaminant modeling