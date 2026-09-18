"""Legacy profile-likelihood routines in star-parameter coordinates.

These routines mutate the fiducial cosmology. They are retained as reference
only; the active inference API no longer exposes profile likelihoods in these
coordinates.
"""

import os
import time

import numpy as np
from mpi4py import MPI

from cup1d.utils import blinding


def rescale_fid_cosmo(theory, target_params):
    """Legacy conversion from star to primordial parameters."""

    fiducial_cosmo = theory.fid_cosmo["cosmo"]
    dkms_dMpc = fiducial_cosmo.get_dkms_dMpc(theory.z_star)
    kp_Mpc = theory.kp_kms * dkms_dMpc
    ks_Mpc = fiducial_cosmo.CAMBparams.InitPower.pivot_scalar
    pstar = theory.fid_cosmo["linP_params"]
    delta_ns = target_params["n_star"] - pstar["n_star"]
    ln_ratio_As = np.log(target_params["Delta2_star"] / pstar["Delta2_star"])
    ln_ratio_As -= delta_ns * np.log(kp_Mpc / ks_Mpc)

    cosmo_params_dict = fiducial_cosmo.input_cosmo_params_dict.copy()
    cosmo_params_dict.update(
        As=np.exp(ln_ratio_As) * fiducial_cosmo.CAMBparams.InitPower.As,
        ns=delta_ns + fiducial_cosmo.CAMBparams.InitPower.ns,
    )
    theory.set_fid_cosmo(
        theory.fid_cosmo["zs"], cosmo_params_dict=cosmo_params_dict
    )


def run_fitter_profile(
    fitter,
    irank,
    mle_cosmo_cen,
    shift_cosmo,
    input_pars,
    type_minimizer="NM",
    verbose=True,
):
    """Legacy single-point profile calculation."""

    blind_cosmo = {
        "Delta2_star": mle_cosmo_cen["Delta2_star"] + shift_cosmo["Delta2_star"],
        "n_star": mle_cosmo_cen["n_star"] + shift_cosmo["n_star"],
    }
    if verbose:
        print("\nStarting profile", irank, blind_cosmo, "\n")

    target = blinding.apply_unblinding(fitter.like.blind, mle_cosmo_cen)
    target["Delta2_star"] += shift_cosmo["Delta2_star"]
    target["n_star"] += shift_cosmo["n_star"]
    rescale_fid_cosmo(fitter.like.theory, target)

    if not np.isfinite(fitter.like.get_chi2(input_pars)):
        print("skipping", irank, blind_cosmo)
        return

    if type_minimizer == "NM":
        fitter.run_minimizer(fitter.like.minus_log_prob, p0=input_pars, restart=True)
    elif type_minimizer == "DA":
        fitter.run_minimizer_da(fitter.like.minus_log_prob, p0=input_pars, restart=True)
    else:
        raise ValueError("type_minimizer must be 'NM' or 'DA'")

    out_dict = {
        "chi2": fitter.mle_chi2,
        "blind_cosmo": blind_cosmo,
        "mle_cube": fitter.mle_cube,
        "mle": fitter.mle,
    }
    file_out = os.path.join(
        os.path.dirname(fitter.save_directory), f"profile_{irank}.npy"
    )
    np.save(file_out, out_dict)


def run_analysis_profile(
    analysis,
    sigma_cosmo,
    mle_cosmo_cen=None,
    nelem=10,
    nsig=10,
    type_minimizer="NM",
    folder_ic=None,
):
    """Legacy grid driver for :func:`run_fitter_profile`."""

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    x = np.linspace(-nsig, nsig, nelem)
    if len(sigma_cosmo) == 1:
        xgrid = sigma_cosmo.get("Delta2_star", 0) * x
        ygrid = sigma_cosmo.get("n_star", 0) * x
    elif len(sigma_cosmo) == 2:
        xgrid, ygrid = np.meshgrid(x, x)
        xgrid = xgrid.reshape(-1) * sigma_cosmo["Delta2_star"]
        ygrid = ygrid.reshape(-1) * sigma_cosmo["n_star"]
    else:
        raise ValueError("sigma_cosmo must contain one or two parameters")

    ind_ranks = np.array_split(np.arange(len(xgrid)), size)
    if rank == 0:
        print("IDs to each rank:", ind_ranks)

    if mle_cosmo_cen is None:
        if rank == 0:
            if folder_ic is None:
                folder_ic = os.path.dirname(os.path.dirname(analysis.fitter.save_directory))
            file_out = os.path.join(folder_ic, "best_dircosmo.npy")
            mle_cosmo_cen = np.load(file_out, allow_pickle=True).item()["mle_cosmo_cen"]
            for task_rank in range(1, size):
                comm.send(mle_cosmo_cen, dest=task_rank, tag=(task_rank + 1) * 5)
        else:
            mle_cosmo_cen = comm.recv(source=0, tag=(rank + 1) * 5)

    pini = analysis.fitter.like.sampling_point_from_parameters().copy()
    if rank == 0:
        start = time.time()
        analysis.fprint("----------")
        analysis.fprint("Running legacy profile likelihood")

    for irank in ind_ranks[rank]:
        run_fitter_profile(
            analysis.fitter,
            irank,
            mle_cosmo_cen,
            {"Delta2_star": xgrid[irank], "n_star": ygrid[irank]},
            pini,
            type_minimizer=type_minimizer,
        )

    if rank == 0:
        analysis.fprint(f"Profile run in {time.time() - start:.2f} s")
        analysis.fprint("----------")
