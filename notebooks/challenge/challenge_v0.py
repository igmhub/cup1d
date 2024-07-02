# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

import numpy as np
import time, os, sys
import matplotlib.pyplot as plt

# our own modules
import lace
from lace.archive import gadget_archive, nyx_archive
from lace.cosmo import camb_cosmo
from lace.emulator.emulator_manager import set_emulator
from cup1d.p1ds import (
    data_gadget,
    data_nyx,
    data_eBOSS_mock,
    data_Chabanier2019,
    data_Karacayli2022,
    data_Karacayli2023,
    data_Ravoux2023,
)
from cup1d.likelihood import lya_theory, likelihood, emcee_sampler
from cup1d.likelihood.sampler_pipeline import set_archive, set_P1D, set_fid_cosmo, set_like
from cup1d.likelihood.input_pipeline import Args

# %%
from cup1d.p1ds.data_QMLE_Ohio import P1D_QMLE_Ohio

# %%
folder = "/home/jchaves/Proyectos/projects/lya/data/cup1d/challenge/MockChallenge-v0.1/"
file = "fiducial_lym1d_p1d_qmleformat_IC.txt"
dat = P1D_QMLE_Ohio(filename = folder+file, z_min=3, z_max=10)
dat.plot_p1d()

# %%
# set output directory for this test
output_dir = "."

# args = Args(emulator_label="Pedersen21")
# args = Args(emulator_label="Cabayol23+", training_set="Cabayol23")
args = Args(emulator_label="Nyx_alphap", training_set="Nyx23_Oct2023")
args.n_igm=1
args.n_steps=50
args.n_burn_in=10
args.parallel=False
args.explore=True
args.data_label="challenge_v0"

# %%
archive = set_archive(args.training_set)

# %%
emulator = set_emulator(
    emulator_label=args.emulator_label,
    archive=archive,
)

# %%
cosmo_fid = set_fid_cosmo(cosmo_label=args.cosmo_label)

# %%
data = {"P1Ds": None, "extra_P1Ds": None}
data["P1Ds"], true_sim_igm = set_P1D(
    archive,
    emulator,
    args.data_label,
    cosmo_fid,
    cov_label=args.cov_label,
    apply_smoothing=False
)

# %%
like = set_like(
    emulator,
    data["P1Ds"],
    data["extra_P1Ds"],
    true_sim_igm,
    args.igm_label,
    args.n_igm,
    cosmo_fid,
)

# %%
# like.plot_p1d(residuals=True, plot_every_iz=2)

# %%
for p in like.free_params:
    print(p.name, p.value, p.min_value, p.max_value)


# %%
def log_prob(theta):
    return log_prob.sampler.like.log_prob_and_blobs(theta)

def set_log_prob(sampler):
    log_prob.sampler = sampler
    return log_prob

sampler = emcee_sampler.EmceeSampler(
    like=like,
    rootdir=output_dir,
    save_chain=False,
    nburnin=args.n_burn_in,
    nsteps=args.n_steps,
    parallel=args.parallel,
    explore=args.explore,
    fix_cosmology=args.fix_cosmo,
)
_log_prob = set_log_prob(sampler)

# %%
# %%time
_emcee_sam = sampler.run_sampler(log_func=_log_prob)

# %%
# %%time
ind = np.argmax(sampler.lnprob.reshape(-1))
nparam = sampler.chain.shape[-1]
p0 = sampler.chain.reshape(-1, nparam)[ind, :]
sampler.run_minimizer(log_func=_log_prob, p0=p0)

# %%
p1 = {'Delta2_p': 0.6424254870204057, 'n_p': -2.284963361317453, 'alpha_p': -0.21536767260941628, 'mF': 0.8333555955907445, 'gamma': 1.5166829814584781, 'sigT_Mpc': 0.10061115435052223, 'kF_Mpc': 10.614589838852988}
k = np.linspace(0.1, 1.7, 50)
z = 2.2
emulator.emulate_p1d_Mpc(p1, k, z=z, return_covar=True)

# %%
sampler.write_chain_to_file()

# %%
