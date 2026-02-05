# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Other figures
#
# Zenodo holder

# +
import cup1d, os
import numpy as np

path_out = os.path.join(os.path.dirname(cup1d.__path__[0]), "data", "zenodo")
store_data = {
    "figs11_to_16":"check out the first two columns of Table 5",
}
fname = os.path.join(path_out, "fig_11_to_16.npy")
np.save(fname, store_data)

# +
import cup1d, os
import numpy as np

path_out = os.path.join(os.path.dirname(cup1d.__path__[0]), "data", "zenodo")
store_data = {
    "figs23_24":"Table 6",
}
fname = os.path.join(path_out, "fig_23_24.npy")
np.save(fname, store_data)

# +
import cup1d, os
import numpy as np

path_out = os.path.join(os.path.dirname(cup1d.__path__[0]), "data", "zenodo")
store_data = {
    "blue":{
        "mu":-0.026,
        "sigma":0.203
    },
    "orange":{
        "A":-0.047,
        "B":0.198
    },
}
fname = os.path.join(path_out, "fig_25.npy")
np.save(fname, store_data)

# +
import cup1d, os
import numpy as np

path_out = os.path.join(os.path.dirname(cup1d.__path__[0]), "data", "zenodo")
store_data = {
    "fig_D1":"Table D1",
}
fname = os.path.join(path_out, "fig_D1.npy")
np.save(fname, store_data)

# +
import cup1d, os

path_out = os.path.join(os.path.dirname(cup1d.__path__[0]), "data", "zenodo")
fname = os.path.join(path_out, "other_figures.npy")
store_data = {
    "figs11_to_16":"first two columns of Table 5",
    "fig18":"likelihoods in https://github.com/igmhub/cobaya_lya_p1d or check out corresponding papers",
    "figs23_24":"Table 6",
    "figD1": "Table D1",
}
np.save(fname, store_data)
