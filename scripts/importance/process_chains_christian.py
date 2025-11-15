import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"  # export OMP_NUM_THREADS=4
import numpy as np
from mpi4py import MPI
from cup1d.planck import planck_chains
from cup1d.planck import add_linP_params
from cup1d.utils.utils import get_path_repo


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    key_model = "base_mnu"
    key_data = "desi-bao-all_planck2018-lowl-TT-clik_planck2018-lowl-EE-clik_planck-NPIPE-highl-CamSpec-TTTEEE_planck-act-dr6-lensing"

    root_dir = os.path.join(
        get_path_repo("cup1d"), "data", "planck_linP_chains", "crisjagq"
    )

    if rank == 0:
        cmb = planck_chains.get_cobaya(
            root_dir=root_dir,
            model=key_model,
            data=key_data,
            linP_tag=None,
        )

        samples = cmb["samples"]
        # thinning = 4000
        # samples.thin(thinning)
        Nsamp, Npar = samples.samples.shape
        print(Nsamp, Npar, flush=True)
        params = []
        for ii in range(Nsamp):
            params.append(cmb["samples"].getParamSampleDict(ii))

        ind_ranks = np.array_split(np.arange(len(params)), size)

        _ind = slice(ind_ranks[rank][0], ind_ranks[rank][-1] + 1)
        _params = params[_ind]
        for irank in range(1, size):
            _ind = slice(ind_ranks[irank][0], ind_ranks[irank][-1] + 1)
            comm.send(params[_ind], dest=irank, tag=irank * 3)
    else:
        _params = comm.recv(source=0, tag=rank * 3)

    _Nsamp = len(_params)
    linP_entries = np.zeros((_Nsamp, 2))
    # linP_entries = np.zeros((_Nsamp, 4))
    for ii in range(_Nsamp):
        # get point from original chain
        # compute linear power parameters (n_star, f_star, etc.)
        linP_params1 = add_linP_params.get_linP_params(
            _params[ii], z_star=3.0, kp_kms=0.009, camb_kmax_Mpc_fast=1.5
        )

        # linP_params2 = add_linP_params.get_linP_params(
        #     _params[ii],
        #     z_star=3.0,
        #     kp_kms=0.023,
        #     camb_kmax_Mpc_fast=10.0,
        # )

        linP_entries[ii, 0] = linP_params1["Delta2_star"]
        linP_entries[ii, 1] = linP_params1["n_star"]
        # linP_entries[ii, 2] = linP_params2["Delta2_star"]
        # linP_entries[ii, 3] = linP_params2["n_star"]

        if (rank == 0) and (ii % 10 == 0):
            print("sample point", ii, "out of", _Nsamp, flush=True)
            print("linP params", linP_entries[ii], flush=True)

    if rank == 0:
        linP_DL2_star = np.zeros(Nsamp)
        linP_n_star = np.zeros(Nsamp)
        # linP_DL2_star2 = np.zeros(Nsamp)
        # linP_n_star2 = np.zeros(Nsamp)

        linP_DL2_star[ind_ranks[0]] = linP_entries[:, 0]
        linP_n_star[ind_ranks[0]] = linP_entries[:, 1]
        # linP_DL2_star2[ind_ranks[0]] = linP_entries[:, 2]
        # linP_n_star2[ind_ranks[0]] = linP_entries[:, 3]
        for irank in range(1, size):
            _linP_entries = comm.recv(source=irank, tag=irank * 5)
            linP_DL2_star[ind_ranks[irank]] = _linP_entries[:, 0]
            linP_n_star[ind_ranks[irank]] = _linP_entries[:, 1]
            # linP_DL2_star2[ind_ranks[irank]] = _linP_entries[:, 2]
            # linP_n_star2[ind_ranks[irank]] = _linP_entries[:, 3]
    else:
        comm.send(linP_entries, dest=0, tag=rank * 5)

    if rank == 0:
        samples.addDerived(
            linP_DL2_star, "linP_DL2_star", label="Ly\\alpha \\, \\Delta_\\ast"
        )
        samples.addDerived(
            linP_n_star, "linP_n_star", label="Ly\\alpha \\, n_\\ast"
        )
        # samples.addDerived(
        #     linP_DL2_star2,
        #     "linP_DL2_star2",
        #     label="Ly\\alpha \\, \\Delta_\\ast2",
        # )
        # samples.addDerived(
        #     linP_n_star2, "linP_n_star2", label="Ly\\alpha \\, n_\\ast2"
        # )

        new_root = os.path.join(
            root_dir,
            key_model,
            key_data + "_linP",
            key_model + "_" + key_data + "_linP",
        )
        print("Saving to", new_root, flush=True)
        samples.saveAsText(root=new_root, make_dirs=True)


if __name__ == "__main__":
    main()
