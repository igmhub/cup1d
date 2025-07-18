import re

import numpy as np
from cup1d.likelihood.pipeline import set_like
from cup1d.likelihood.input_pipeline import Args
from cup1d.likelihood.fitter import Fitter
from cup1d.optimize.show_results import get_parameters


def split_string(s):
    match = re.match(r"^(.*)_(\d+)$", s)
    if match:
        return match.group(1), match.group(2)
    else:
        return s, None


def set_ic_from_fullfit(
    like,
    data,
    emulator,
    fname,
    type_fit="global",
    output_dir=".",
    verbose=True,
):
    """Set the initial conditions for the likelihood from a fit"""

    args = Args(emulator_label="CH24_mpgcen_gpr", training_set="Cabayol23")
    args.set_baseline(fit_type=type_fit, fix_cosmo=False, zmax=4.2)

    dir_out = np.load(fname, allow_pickle=True).item()

    like1 = set_like(
        data["P1Ds"],
        emulator,
        args,
        data_hires=data["extra_P1Ds"],
    )
    out_mle_cube_reformat = dir_out["mle_cube"]

    # best-fitting star params of full fit (blinded). for profiling, move around them
    mle_cosmo = {}
    for key in ["Delta2_star", "n_star", "alpha_star"]:
        mle_cosmo[key] = dir_out["mle"][key]

    # make a copy of free params, and set their values to the best-fit
    free_params = like.free_params.copy()
    for jj, p in enumerate(free_params):
        if p.name in ["As", "ns"]:
            continue
        pname, iistr = split_string(p.name)
        ii = int(iistr)

        if (pname + "_znodes") in args.fid_igm:
            znode = args.fid_igm[pname + "_znodes"][ii]
        else:
            znode = args.fid_cont[pname + "_znodes"][ii]

        ind = np.argwhere(p.name == np.array(like1.free_param_names))[0, 0]
        p.value = like1.free_params[ind].value_from_cube(
            out_mle_cube_reformat[ind]
        )

        if verbose:
            print(
                p.name,
                "\t",
                np.round(p.value, 3),
                "\t",
                np.round(p.min_value, 3),
                "\t",
                np.round(p.max_value, 3),
                "\t",
                p.Gauss_priors_width,
                p.fixed,
            )

    # reset the coefficients of the models
    like.theory.model_igm.models["F_model"].reset_coeffs(free_params)
    like.theory.model_igm.models["T_model"].reset_coeffs(free_params)
    like.theory.model_cont.hcd_model.reset_coeffs(free_params)
    like.theory.model_cont.metal_models["Si_mult"].reset_coeffs(free_params)
    like.theory.model_cont.metal_models["Si_add"].reset_coeffs(free_params)

    args.n_steps = 5
    args.n_burn_in = 1
    args.parallel = False
    args.explore = True

    fitter = Fitter(
        like=like,
        rootdir=output_dir,
        nburnin=args.n_burn_in,
        nsteps=args.n_steps,
        parallel=args.parallel,
        explore=args.explore,
        fix_cosmology=args.fix_cosmo,
    )

    # rescale the fiducial cosmology
    tar = fitter.apply_unblinding(mle_cosmo)
    fitter.like.theory.rescale_fid_cosmo(tar)

    return fitter


def set_ic_from_z_at_time(
    args,
    like,
    data,
    emulator,
    fname,
    output_dir=".",
    verbose=True,
):
    """Set the initial conditions for the likelihood from a fit"""

    args2 = Args(emulator_label="CH24_mpgcen_gpr", training_set="Cabayol23")
    args2.set_baseline(ztar=data["P1Ds"].z[0], fit_type="at_a_time")

    dir_out = np.load(fname, allow_pickle=True).item()

    like1 = set_like(
        data["P1Ds"],
        emulator,
        args2,
        data_hires=data["extra_P1Ds"],
    )
    out_mle_cube_reformat = dir_out["mle_cube_reformat"]

    # make a copy of free params, and set their values to the best-fit
    free_params = like.free_params.copy()
    for jj, p in enumerate(free_params):
        if p.name in ["As", "ns"]:
            continue
        pname, iistr = split_string(p.name)
        ii = int(iistr)

        if (pname + "_znodes") in args.fid_igm:
            znode = args.fid_igm[pname + "_znodes"][ii]
        else:
            znode = args.fid_cont[pname + "_znodes"][ii]

        iz = np.argmin(np.abs(like1.data.z - znode))
        p.value = get_parameters(pname, znode, like1, out_mle_cube_reformat[iz])

        if verbose:
            print(
                p.name,
                "\t",
                np.round(p.value, 3),
                "\t",
                np.round(p.min_value, 3),
                "\t",
                np.round(p.max_value, 3),
                "\t",
                p.Gauss_priors_width,
                p.fixed,
            )

    # reset the coefficients of the models
    like.theory.model_igm.models["F_model"].reset_coeffs(free_params)
    like.theory.model_igm.models["T_model"].reset_coeffs(free_params)
    like.theory.model_cont.hcd_model.reset_coeffs(free_params)
    like.theory.model_cont.metal_models["Si_mult"].reset_coeffs(free_params)
    like.theory.model_cont.metal_models["Si_add"].reset_coeffs(free_params)

    args.n_steps = 5
    args.n_burn_in = 1
    args.parallel = False
    args.explore = True

    fitter = Fitter(
        like=like,
        rootdir=output_dir,
        nburnin=args.n_burn_in,
        nsteps=args.n_steps,
        parallel=args.parallel,
        explore=args.explore,
        fix_cosmology=args.fix_cosmo,
    )

    return fitter
