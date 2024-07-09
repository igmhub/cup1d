# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
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

# +
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
# -

from cup1d.p1ds.data_QMLE_Ohio import P1D_QMLE_Ohio

folder = "/home/jchaves/Proyectos/projects/lya/data/cup1d/challenge/MockChallenge-v0.1/"
file = "fiducial_lym1d_p1d_qmleformat_IC.txt"
dat = P1D_QMLE_Ohio(filename = folder+file)
dat.plot_p1d()

# +
# set output directory for this test
output_dir = "."

# args = Args(emulator_label="Pedersen21")
# args = Args(emulator_label="Cabayol23+", training_set="Cabayol23")
# args = Args(emulator_label="Nyx_v0", training_set="Nyx23_Oct2023")
args = Args(emulator_label="Nyx_alphap", training_set="Nyx23_Oct2023")
args.n_igm=1
args.n_steps=50
args.n_burn_in=10
args.parallel=False
args.explore=True
args.data_label="challenge_v0"
args.cosmo_label = "nyx_central"
# -

archive = set_archive(args.training_set)

emulator = set_emulator(
    emulator_label=args.emulator_label,
    archive=archive,
)

cosmo_fid = set_fid_cosmo(cosmo_label=args.cosmo_label)

data = {"P1Ds": None, "extra_P1Ds": None}
data["P1Ds"], true_sim_igm = set_P1D(
    archive,
    emulator,
    args.data_label,
    cosmo_fid,
    cov_label=args.cov_label,
    apply_smoothing=False
)

like = set_like(
    emulator,
    data["P1Ds"],
    data["extra_P1Ds"],
    true_sim_igm,
    args.igm_label,
    args.n_igm,
    cosmo_fid,
)

like.plot_p1d(residuals=True, plot_every_iz=2)

for p in like.free_params:
    print(p.name, p.value, p.min_value, p.max_value)


# +
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
# -

# %%time
_emcee_sam = sampler.run_sampler(log_func=_log_prob)

# %%time
ind = np.argmax(sampler.lnprob.reshape(-1))
nparam = sampler.chain.shape[-1]
p0 = sampler.chain.reshape(-1, nparam)[ind, :]
sampler.run_minimizer(log_func=_log_prob, p0=p0)

sampler.write_chain_to_file()



# ## Results

from corner import corner

folder = os.environ["CHALLENGE_PATH"] + "chain_3/"
lnprob = np.load(folder + "lnprob.npy")
chain = np.load(folder + "chain.npy")
chain.shape

lnprob_min = 40
mask = np.argwhere(lnprob>lnprob_min)[:,0]
# plt.hist(lnprob[lnprob<lnprob_min], bins=100);

# +

# Delta2_star= 0.33858207513696525
# n_star= -2.3139639485226837
# corner(chain[mask][:, -2:], range=([0.305, 0.35], [-2.332, -2.315]), truths=[Delta2_star, n_star]);

corner(chain[mask][:, -2:], range=([0.305, 0.35], [-2.332, -2.315]), labels=["Delta2_star", "n_star"]);
plt.savefig("corner_challenge_v0.png")
# -

# Initial conditions

fiducial_params = {
    "2.0": {
        "Delta2_p": 0.5915481739911567,
        "n_p": -2.3139647217213906,
        "mF": 0.8769945913550308,
        "sigma_T": 0.13618423774228078,
        "gamma": 1.4555612136788951,
        "kF_Mpc": 14.356000234947775,
    },
    "2.2": {
        "Delta2_p": 0.5226207776924,
        "n_p": -2.3139638065244412,
        "mF": 0.8472177619488307,
        "sigma_T": 0.13277473855710425,
        "gamma": 1.4555612136788951,
        "kF_Mpc": 14.356000234947775,
    },
    "2.4": {
        "Delta2_p": 0.46484410421333394,
        "n_p": -2.3139650206935016,
        "mF": 0.8134383551305827,
        "sigma_T": 0.1295142773903417,
        "gamma": 1.4555612136788951,
        "kF_Mpc": 14.356000234947775,
    },
    "2.6": {
        "Delta2_p": 0.41598979457256346,
        "n_p": -2.313964038190875,
        "mF": 0.7757297063850016,
        "sigma_T": 0.1264136485148265,
        "gamma": 1.4555612136788951,
        "kF_Mpc": 14.356000234947775,
    },
    "2.8": {
        "Delta2_p": 0.3743455589310739,
        "n_p": -2.3139648899314125,
        "mF": 0.7342887533400222,
        "sigma_T": 0.12347433619063276,
        "gamma": 1.4555612136788951,
        "kF_Mpc": 14.356000234947775,
    },
    "3.0": {
        "Delta2_p": 0.33858207513696525,
        "n_p": -2.3139639485226837,
        "mF": 0.6894431961710519,
        "sigma_T": 0.12069245499592879,
        "gamma": 1.4555612136788951,
        "kF_Mpc": 14.356000234947775,
    },
    "3.2": {
        "Delta2_p": 0.30765729829888805,
        "n_p": -2.3139633880766723,
        "mF": 0.6416525729939673,
        "sigma_T": 0.11806116900119289,
        "gamma": 1.4555612136788951,
        "kF_Mpc": 14.356000234947775,
    },
    "3.4": {
        "Delta2_p": 0.28074666500606277,
        "n_p": -2.313964236754373,
        "mF": 0.5915020921117724,
        "sigma_T": 0.11557217089103342,
        "gamma": 1.4555612136788951,
        "kF_Mpc": 14.356000234947775,
    },
    "3.6": {
        "Delta2_p": 0.2571914938794015,
        "n_p": -2.3139640454277526,
        "mF": 0.5396884735521884,
        "sigma_T": 0.11321657852544513,
        "gamma": 1.4555612136788951,
        "kF_Mpc": 14.356000234947775,
    },
    "3.8": {
        "Delta2_p": 0.23646151036900714,
        "n_p": -2.3139638694833593,
        "mF": 0.48699763886263747,
        "sigma_T": 0.11098546951141154,
        "gamma": 1.4555612136788951,
        "kF_Mpc": 14.356000234947775,
    },
    "4.0": {
        "Delta2_p": 0.21812586239469972,
        "n_p": -2.31396410289733,
        "mF": 0.4342748177846472,
        "sigma_T": 0.10887019053694555,
        "gamma": 1.4555612136788951,
        "kF_Mpc": 14.356000234947775,
    },
    "4.2": {
        "Delta2_p": 0.2018322959857999,
        "n_p": -2.313964614946059,
        "mF": 0.3823884497673086,
        "sigma_T": 0.10686252671587962,
        "gamma": 1.4555612136788951,
        "kF_Mpc": 14.356000234947775,
    },
    "4.4": {
        "Delta2_p": 0.1872902381389085,
        "n_p": -2.3139645298519724,
        "mF": 0.3321900554898156,
        "sigma_T": 0.10495478435982362,
        "gamma": 1.4555612136788951,
        "kF_Mpc": 14.356000234947775,
    },
}

# dat = archive.get_testing_data("nyx_central")
p1 = {'Delta2_p': 0.6424254870204057, 'n_p': -2.284963361317453, 'alpha_p': -0.21536767260941628, 'mF': 0.8333555955907445, 'gamma': 1.5166829814584781, 'sigT_Mpc': 0.10061115435052223, 'kF_Mpc': 10.614589838852988}
k = np.linspace(0.2, 1.7, 50)
z = 2.2
emulator.emulate_p1d_Mpc(p1, k, z=z, return_covar=False)

# +
from lace.cosmo.camb_cosmo import get_Nyx_cosmology, dkms_dMpc

cosmo_params = {}
cosmo_params["H_0"] = 67.78216034931903
cosmo_params["omega_m"] = 0.1398651784897052
cosmo_params["A_s"] = 1e-9
cosmo_params["n_s"] = 0.

cosmo_fid = get_Nyx_cosmology(cosmo_params)

for key in fiducial_params.keys():
    z = float(key)
    if(z > 2):
        p1 = fiducial_params[key]
        p1["alpha_p"] = -0.21538810074866552
        p1["sigT_Mpc"] = p1["sigma_T"]
        _ = np.argwhere(dat.z == z)[0, 0]
        _dkms_dMpc = dkms_dMpc(cosmo_fid, z)
        k = dat.k_kms[_] * _dkms_dMpc
        plt.plot(k, dat.Pk_kms[_]*dat.k_kms[_]/np.pi, label="data")

        pk_emu = emulator.emulate_p1d_Mpc(p1, k, z=z, return_covar=False)
        plt.plot(k, pk_emu*k/np.pi, label="emu")
        plt.legend()
        plt.xlabel("k[Mpc]")
        plt.ylabel("pk*k/pi")
        break
plt.savefig("test.png")
