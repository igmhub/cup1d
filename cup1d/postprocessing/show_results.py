import numpy as np

from cup1d.likelihood import parameter as parameter_space
from scipy.stats import chi2 as chi2_scipy


def get_parameters(par, z, like, mle_cube):
    """Evaluate one fitted IGM or contaminant parameter at a redshift.

    Parameters
    ----------
    par : str
        Coefficient name registered by an IGM, metal, or HCD model.
    z : float or array_like
        Redshift or redshifts at which to evaluate the model.
    like : cup1d.likelihood.likelihood.Likelihood
        Likelihood supplying free-parameter metadata and fitted models.
    mle_cube : array_like
        Best-fit point in normalized sampling-cube coordinates.

    Returns
    -------
    float or ndarray
        Physical model value. Values of exponentially parameterized models are
        returned in natural-log form for consistency with their coefficients.

    Raises
    ------
    ValueError
        If ``par`` is not provided by the supported fitted models.
    """
    like_params = parameter_space.values_from_cube(like.free_params, mle_cube)

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
    """Map per-redshift at-a-time best fits into a common sampling cube.

    Parameters
    ----------
    args : cup1d.inference.args.Args
        Analysis configuration. Its baseline is updated in place for each
        P1D redshift bin.
    data : mapping
        Input data with ``"P1Ds"`` and ``"extra_P1Ds"`` entries used to build
        the at-a-time likelihoods.
    emulator : object
        Emulator passed to :func:`cup1d.inference.analysis.set_like`.
    out_mle_cube : sequence of array_like
        One normalized maximum-likelihood cube per P1D redshift bin.
    weak_priors : mapping, optional
        Redshift-indexed weak-prior centers and widths for free parameters.

    Returns
    -------
    list of ndarray
        At-a-time best-fit points represented in the free-parameter ordering
        of the first redshift-bin likelihood.
    """
    from cup1d.inference.analysis import set_like

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
            for name, parameter in like2.free_params.items():
                if name not in list_fix:
                    parameter["value"] = weak_priors[name + "_cen"][ii]
                    parameter["min_value"] = (
                        weak_priors[name + "_cen"][ii]
                        - 2 * weak_priors[name + "_std"]
                    )
                    parameter["max_value"] = (
                        weak_priors[name + "_cen"][ii]
                        + 2 * weak_priors[name + "_std"]
                    )
                else:
                    if (parameter["value"] < parameter["max_value"]) & (
                        parameter["value"] > parameter["min_value"]
                    ):
                        parameter["value"] = weak_priors[name + "_cen"][ii]

        _cube = np.zeros(len(like1.free_param_names))
        for jj, prop in enumerate(like1.free_param_names):
            if prop in like2.free_param_names:
                ind = np.argwhere(prop == np.array(like2.free_param_names))[
                    0, 0
                ]
                value = parameter_space.value_from_cube(like2.free_params, prop, out_mle_cube[ii][ind])
                in_cube = parameter_space.value_in_cube(like1.free_params, prop, value)
                print(prop)
                if in_cube < 0:
                    in_cube = 0
                _cube[jj] = in_cube
        out_mle_cube_reformat.append(np.array(_cube))

    return out_mle_cube_reformat


def print_results(like, out_chi2, out_mle_cube):
    """Print goodness-of-fit statistics for independent redshift-bin fits.

    Parameters
    ----------
    like : cup1d.likelihood.likelihood.Likelihood
        Likelihood containing exactly one P1D data set.
    out_chi2 : sequence of float
        Best-fit chi-squared values, ordered by P1D redshift bin.
    out_mle_cube : sequence of array_like
        Best-fit sampling cubes aligned with ``out_chi2``. Their lengths set
        the number of fitted degrees of freedom per bin.

    Raises
    ------
    ValueError
        If the likelihood contains multiple P1D data sets or more fit results
        than redshift bins.
    """

    if len(like.data) != 1:
        raise ValueError("print_results requires exactly one P1D data set")
    data = next(iter(like.data.values()))
    if len(out_chi2) > len(data.z):
        raise ValueError("More fit results than available P1D redshift bins")

    ndeg_all = 0
    props = []
    chi2_all = 0
    print(r"$z$ & $\chi^2$ & ndeg & prob\\ \hline")
    for ii in range(len(out_chi2)):
        ndeg = len(data.k_kms[ii]) - len(out_mle_cube[ii])
        prob = chi2_scipy.sf(out_chi2[ii], ndeg)
        print(
            data.z[ii],
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
    print(r"\hline")
    print(
        "All",
        "&",
        np.round(chi2_all, 2),
        "&",
        ndeg_all,
        "&",
        np.round(prob * 100, 2),
        "\\\\",
        r"\hline",
    )
    print("Prob", prob * 100)
