Most Planck chains are not provided in cup1d, but they can be downloaded from
the official Planck website: https://wiki.cosmos.esa.int/planck-legacy-archive/index.php/Cosmological_Parameters

We do provide a handful of chains under `$CUP1D_PATH/planck_linP_chains/`.

These chains have extra columns with the linear-power parameters relevant for
Lyman-alpha P1D: `linP_DL2_star`, `linP_n_star`, `linP_alpha_star`,
`linP_f_star`, and `linP_g_star`.

`planck.load_planck_2018_chains` loads a named collection of Planck chains
from a list of model/data specifications. `planck.load_spa_chains` provides
the corresponding interface for CMB-SPA chains. Use these helpers when a
notebook or plot needs several chains, rather than duplicating individual
loading calls.
