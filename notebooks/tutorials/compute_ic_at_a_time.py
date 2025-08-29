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

# # Compute IC from fits at a time

# +
# %load_ext autoreload
# %autoreload 2

import numpy as np
import time, os, sys
import matplotlib.pyplot as plt

# our own modules
from cup1d.likelihood.input_pipeline import Args
from cup1d.likelihood.pipeline import Pipeline
from cup1d.likelihood.plotter import Plotter
from cup1d.utils.utils import get_path_repo
# -


emu = "mpg"
args = Args(data_label="DESIY1_QMLE3", emulator_label="CH24_"+emu+"cen_gpr")
args.set_baseline(fit_type="at_a_time_global", fix_cosmo=True)
pip = Pipeline(args, out_folder=None)

# ### Do fits

out_mle = []
out_mle_cube = []
out_chi2 = []
out_pnames = []
# for ii in range(len(pip.fitter.like.data.z)):
for ii in range(1):
    zmask = np.array([pip.fitter.like.data.z[ii]])

    pip = Pipeline(args, out_folder=None)
    
    print()
    
    f_space_len = 14
    s_space_len = 5
    for p in pip.fitter.like.free_params:            
        print(
            p.name, (f_space_len-len(p.name)) * " ", "\t", 
            np.round(p.value, 3), (s_space_len-len(str(np.round(p.value, 3)))) * " ", '\t', 
            np.round(p.min_value, 3), (s_space_len-len(str(np.round(p.min_value, 3)))) * " ", '\t', 
            np.round(p.max_value, 3), (s_space_len-len(str(np.round(p.max_value, 3)))) * " ", '\t', 
            p.Gauss_priors_width
        )

    
    print()
    
    print(ii, zmask)
    p0 = np.array(list(pip.fitter.like.fid["fit_cube"].values()))
    pip.fitter.run_minimizer(log_func_minimize=pip.fitter.like.minus_log_prob, p0=p0, zmask=zmask, restart=True)
    out_pnames.append(pip.fitter.like.free_param_names)
    out_mle.append(pip.fitter.mle)
    out_mle_cube.append(pip.fitter.mle_cube)
    out_chi2.append(pip.fitter.mle_chi2)

diru = 'figs'
plotter = Plotter(pip.fitter, save_directory=diru, zmask=zmask)

# +

plotter.plot_illustrate_contaminants_cum(out_mle_cube[0].copy(), zmask, fontsize=20)
# -

fname = os.path.join(
    os.path.dirname(get_path_repo("cup1d")), "data", "ics", emu + "_ic_at_a_time.npy"
)
dir_out = {
    "z":pip.fitter.like.data.z,
    "pnames":out_pnames,
    "mle_cube":out_mle_cube,
    "mle":out_mle,
    "chi2":out_chi2,
}
np.save(fname, dir_out)

from cup1d.optimize.show_results import print_results
print_results(pip.fitter.like, out_chi2, out_mle_cube)

# mpg
$z$ & $\chi^2$ & ndeg & prob\ \hline
2.2 & 29.91 & 36 & 75.25 \\
2.4 & 43.87 & 39 & 27.26 \\
2.6 & 55.84 & 42 & 7.48 \\
2.8 & 51.64 & 45 & 23.03 \\
3.0 & 70.98 & 48 & 1.72 \\
3.2 & 53.91 & 50 & 32.72 \\
3.4 & 48.28 & 52 & 62.09 \\
3.6 & 83.18 & 54 & 0.66 \\
3.8 & 62.54 & 56 & 25.52 \\
4.0 & 81.48 & 57 & 1.84 \\
4.2 & 64.47 & 59 & 29.11 \\
\hline
All & 646.12 & 538 & 0.09 \\ \hline
Prob 0.09105993734928322

# nyx
$z$ & $\chi^2$ & ndeg & prob\ \hline
2.2 & 29.88 & 36 & 75.4 \\
2.4 & 42.52 & 39 & 32.2 \\
2.6 & 55.87 & 42 & 7.44 \\
2.8 & 55.56 & 45 & 13.45 \\
3.0 & 69.56 & 48 & 2.26 \\
3.2 & 53.78 & 50 & 33.18 \\
3.4 & 46.49 & 52 & 68.97 \\
3.6 & 83.07 & 54 & 0.67 \\
3.8 & 62.3 & 56 & 26.21 \\
4.0 & 81.28 & 57 & 1.9 \\
4.2 & 65.49 & 59 & 26.19 \\
\hline
All & 645.78 & 538 & 0.09 \\ \hline
Prob 0.09387017518663274
