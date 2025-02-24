# read emcee chains and get them ready to plot with getdist
import numpy as np
from getdist import MCSamples
from cup1d.likelihood import fitter


# for each parameter name, figure out LaTeX label
param_latex_dict = {
    "Delta2_star": "\Delta^2_\star",
    "n_star": "n_\star",
    "alpha_star": "\\alpha_\star",
    "g_star": "g_\star",
    "f_star": "f_\star",
    "ln_tau_0": "\mathrm{ln} \\tau_0",
    "ln_tau_1": "\mathrm{ln} \\tau_1",
    "ln_tau_2": "\mathrm{ln} \\tau_2",
    "ln_tau_3": "\mathrm{ln} \\tau_3",
    "ln_tau_4": "\mathrm{ln} \\tau_4",
    "ln_sigT_kms_0": "\mathrm{ln} \\sigma^T_0",
    "ln_sigT_kms_1": "\mathrm{ln} \\sigma^T_1",
    "ln_sigT_kms_2": "\mathrm{ln} \\sigma^T_2",
    "ln_sigT_kms_3": "\mathrm{ln} \\sigma^T_3",
    "ln_sigT_kms_4": "\mathrm{ln} \\sigma^T_4",
    "ln_gamma_0": "\mathrm{ln} \\gamma_0",
    "ln_gamma_1": "\mathrm{ln} \\gamma_1",
    "ln_gamma_2": "\mathrm{ln} \\gamma_2",
    "ln_gamma_3": "\mathrm{ln} \\gamma_3",
    "ln_gamma_4": "\mathrm{ln} \\gamma_4",
    "ln_kF_0": "\mathrm{ln} k^F_0",
    "ln_kF_1": "\mathrm{ln} k^F_1",
    "ln_kF_2": "\mathrm{ln} k^F_2",
    "ln_kF_3": "\mathrm{ln} k^F_3",
    "ln_kF_4": "\mathrm{ln} k^F_4",
    "H0": "H_0",
    "mnu": "\\Sigma m_{\\nu}",
    "As": "A_s",
    "ns": "n_s",
    "nrun": "\\alpha_s",
    "ombh2": "\omega_b",
    "omch2": "\omega_c",
    "cosmomc_theta": "\theta_{MC}",
    "lnprob": "log(prob)",
}


def read_chain_for_getdist(
    rootdir,
    subfolder,
    chain_num,
    label,
    delta_lnprob_cut=50,
    ignore_rows=0.2,
    smooth_scale=0.2,
):
    print("will read chain for", label, rootdir, subfolder, chain_num)
    run = {"chain_num": chain_num, "label": label}
    sampler = fitter.EmceeSampler(
        read_chain_file=chain_num, rootdir=rootdir, subfolder=subfolder
    )
    run["sampler"] = sampler

    print("figure out free parameters for", label)
    param_names = [param.name for param in sampler.like.free_params]
    if "n_star" in param_names:
        print("sampling compressed parameters")
        blob_names = None
    else:
        print("sampling cosmo parameters")
        blob_names = ["Delta2_star", "n_star", "alpha_star", "f_star", "g_star"]
        if "H0" not in param_names:
            blob_names.append("H0")

    # read value of free and derived parameters
    free_values, lnprob, blobs = sampler.get_chain(
        cube=False, delta_lnprob_cut=delta_lnprob_cut
    )
    if blob_names:
        blob_values = np.array([blobs[key] for key in blob_names]).transpose()
        # stack all values, including log(prob)
        run["values"] = np.hstack(
            [free_values, np.column_stack((blob_values, lnprob))]
        )
    else:
        run["values"] = np.column_stack((free_values, lnprob))

    # stack all parameter names, including log(prob)
    if blob_names:
        param_names += blob_names
    param_names.append("lnprob")
    run["param_names"] = param_names
    print(label, param_names)
    run["param_labels"] = [param_latex_dict[par] for par in param_names]

    # figure out range of allowed values
    ranges = {}
    for par in sampler.like.free_params:
        ranges[par.name] = [par.min_value, par.max_value]

    # setup getdist object
    samples = MCSamples(
        samples=run["values"],
        label=run["label"],
        names=run["param_names"],
        labels=run["param_labels"],
        ranges=ranges,
        settings={
            "ignore_rows": ignore_rows,
            "mult_bias_correction_order": 0,
            "smooth_scale_2D": smooth_scale,
            "smooth_scale_1D": smooth_scale,
        },
    )
    run["samples"] = samples
    print(label, samples.numrows)

    return run
