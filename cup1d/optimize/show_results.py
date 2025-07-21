import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2 as chi2_scipy


def get_parameters(par, z, like, mle_cube):
    like_params = like.parameters_from_sampling_point(mle_cube)

    models = [
        like.theory.model_igm.models["F_model"],
        like.theory.model_igm.models["T_model"],
        like.theory.model_cont.metal_models["Si_mult"],
        like.theory.model_cont.metal_models["Si_add"],
        like.theory.model_cont.hcd_model,
    ]

    for model in models:
        if par in model.list_coeffs:
            res = model.get_value(par, z, like_params=like_params)
            if model.prop_coeffs[par + "_otype"] == "exp":
                res = np.log(res)
            return res

    raise ValueError(f"Parameter {par} not found")


def reformat_cube(args, data, emulator, out_mle_cube, weak_priors=None):
    from cup1d.likelihood.pipeline import set_like

    ii = 0
    args.set_baseline(ztar=data["P1Ds"].z[ii], fit_type="at_a_time")
    like1 = set_like(
        data["P1Ds"],
        emulator,
        args,
        data_hires=data["extra_P1Ds"],
    )

    out_mle_cube_reformat = []
    for ii in range(len(data["P1Ds"].z)):
        args.set_baseline(ztar=data["P1Ds"].z[ii], fit_type="at_a_time")
        like2 = set_like(
            data["P1Ds"],
            emulator,
            args,
            data_hires=data["extra_P1Ds"],
        )

        if weak_priors is not None:
            for par in like2.free_params:
                if par.name not in list_fix:
                    par.value = weak_priors[par.name + "_cen"][ii]
                    par.min_value = (
                        weak_priors[par.name + "_cen"][ii]
                        - 2 * weak_priors[par.name + "_std"]
                    )
                    par.max_value = (
                        weak_priors[par.name + "_cen"][ii]
                        + 2 * weak_priors[par.name + "_std"]
                    )
                else:
                    if (par.value < par.max_value) & (
                        par.value > par.min_value
                    ):
                        par.value = weak_priors[par.name + "_cen"][ii]

        _cube = np.zeros(len(like1.free_param_names))
        for jj, prop in enumerate(like1.free_param_names):
            if prop in like2.free_param_names:
                ind = np.argwhere(prop == np.array(like2.free_param_names))[
                    0, 0
                ]
                value = like2.free_params[ind].value_from_cube(
                    out_mle_cube[ii][ind]
                )
                in_cube = like1.free_params[jj].get_value_in_cube(value)
                print(prop, like1.free_params[jj].name)
                if in_cube < 0:
                    in_cube = 0
                _cube[jj] = in_cube
        out_mle_cube_reformat.append(np.array(_cube))

    return out_mle_cube_reformat


def print_results(like, out_chi2, out_mle_cube):
    ndeg_all = 0
    props = []
    chi2_all = 0
    print("$z$ & $\chi^2$ & ndeg & prob\\ \hline")
    for ii in range(len(out_chi2)):
        ndeg = len(like.data.k_kms[ii]) - len(out_mle_cube[ii])
        prob = chi2_scipy.sf(out_chi2[ii], ndeg)
        print(
            like.data.z[ii],
            "&",
            np.round(out_chi2[ii], 2),
            "&",
            ndeg,
            "&",
            np.round(prob * 100, 2),
            "\\\\",
        )
        ndeg_all += ndeg
        chi2_all += out_chi2[ii]
        props.append(prob)

    prob = chi2_scipy.sf(chi2_all, ndeg_all)
    print("\hline")
    print(
        "All",
        "&",
        np.round(chi2_all, 2),
        "&",
        ndeg_all,
        "&",
        np.round(prob * 100, 2),
        "\\\\",
        "\hline",
    )
    print("Prob", prob * 100)
