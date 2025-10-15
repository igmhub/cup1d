import matplotlib.pyplot as plt
from cup1d.likelihood.CAMB_model import CAMBModel
from cup1d.likelihood.cosmologies import set_cosmo
import numpy as np

from lace.cosmo import camb_cosmo


# grid of As and ns values without priors
nn = 10
As = np.linspace(1.026970414602751e-09, 3.4382747625135757e-09, nn)
ns = np.linspace(0.6922930599534987, 1.2746555165892985, nn)
A_grid, n_grid = np.meshgrid(As, ns)
A_grid2 = A_grid.reshape(-1)
n_grid2 = n_grid.reshape(-1)
# plt.scatter(A_grid2, n_grid2)

# get corresponding compressed parameters

pl_cosmo = camb_cosmo.get_cosmology(
    H0=67.66,
    mnu=0.0,
    omch2=0.119,
    ombh2=0.0224,
    omk=0.0,
    As=2.105e-09,
    ns=0.9665,
    nrun=0.0,
    pivot_scalar=0.05,
    w=-1,
)
cmodel = CAMBModel([3], pl_cosmo)
res = cmodel.get_linP_params()
pl_dstar = res["Delta2_star"]
pl_nstar = res["n_star"]

mm = A_grid2.shape[0]
dstar = np.zeros(mm)
nstar = np.zeros(mm)
for ii in range(mm):
    # print(A_grid2[ii], n_grid2[ii])
    _cosmo = camb_cosmo.get_cosmology(
        H0=67.66,
        mnu=0.0,
        omch2=0.119,
        ombh2=0.0224,
        omk=0.0,
        As=A_grid2[ii],
        ns=n_grid2[ii],
        nrun=0.0,
        pivot_scalar=0.05,
        w=-1,
    )
    cmodel = CAMBModel([3], _cosmo)
    res = cmodel.get_linP_params()
    # print(res)
    dstar[ii] = res["Delta2_star"]
    nstar[ii] = res["n_star"]

cosmos = set_cosmo(
    cosmo_label="mpg_central",
    return_all=True,
)
for key in cosmos:
    plt.scatter(
        cosmos[key]["star_params"]["Delta2_star"],
        cosmos[key]["star_params"]["n_star"],
        color="red",
    )

plt.scatter(dstar, nstar)
plt.show()

print(dstar.min(), dstar.max())
print(nstar.min(), nstar.max())
