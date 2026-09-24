"""Igm plotting implementations; scientific objects retain compatibility wrappers."""

import numpy as np

from cup1d.likelihood import parameter as parameter_space
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import MaxNLocator
from cup1d.utils.utils import get_path_repo, is_number_string


def plot_likelihood_igm(
    self,
    cloud=False,
    chain_uformat=None,
    free_params=None,
    save_directory=None,
    zmask=None,
    plot_type="all",
    plot_fid=True,
    lab_fid="mpg-central",
    ftsize=18,
    nelem=20000,
    title="",
    pre_xylims=True,
    plot_more_igm=False,
    variation_label="baseline",
    store_data=False,
    plot_external_data=True,
    plot_truth=False,
):
    """Plot IGM histories and optional external measurements or truth."""

    # true IGM parameters
    # if self.truth is not None:
    #     pars_true = {}
    #     pars_true["z"] = self.truth["igm"]["z"]
    #     pars_true["tau_eff"] = self.truth["igm"]["tau_eff"]
    #     pars_true["gamma"] = self.truth["igm"]["gamma"]
    #     pars_true["sigT_kms"] = self.truth["igm"]["sigT_kms"]
    #     pars_true["kF_kms"] = self.truth["igm"]["kF_kms"]

    primary_data = next(iter(self.data.values()))
    zs = np.linspace(primary_data.z.min(), primary_data.z.max(), 100)
    p0 = parameter_space.values_to_cube(self.free_params)

    out = {}
    out["tab_out"] = []
    if store_data:
        out["out_data"] = {}

    for ii in range(3):
        if ii == 0:
            p0[:] = 0.5
        elif ii == 1:
            p0[:] = 0
        elif ii == 2:
            p0[:] = 1
        fid_params = parameter_space.values_from_cube(self.free_params, p0)
        pars = {}
        pars["z"] = zs
        pars["tau_eff"] = self.theory.model_igm.models["F_model"].get_tau_eff(
            zs, like_params=fid_params
        )
        pars["mF"] = self.theory.model_igm.models["F_model"].get_mean_flux(
            zs, like_params=fid_params
        )
        pars["gamma"] = self.theory.model_igm.models["T_model"].get_gamma(
            zs, like_params=fid_params
        )
        pars["sigT_kms"] = self.theory.model_igm.models["T_model"].get_sigT_kms(
            zs, like_params=fid_params
        )
        pars["T0"] = (
            self.theory.model_igm.models["T_model"].get_T0(
                zs, like_params=fid_params
            )
            / 1e4
        )
        pars["kF_kms"] = self.theory.model_igm.models["P_model"].get_kF_kms(
            zs, like_params=fid_params
        )
        if ii == 0:
            pars_fid = pars.copy()
        elif ii == 1:
            pars_min = pars.copy()
        elif ii == 2:
            pars_max = pars.copy()

    chain = None
    if chain_uformat is not None:
        if len(chain_uformat.shape) == 3:
            chain = chain_uformat.reshape(-1, chain_uformat.shape[-1])
        else:
            chain = chain_uformat.copy()
        ind = np.random.permutation(np.arange(0, chain.shape[0]))[:nelem]
        chain = chain[ind]

        if zmask is not None:
            zs2 = zmask
        else:
            zs2 = primary_data.z

        pars_chain = {}
        pars_chain["z"] = zs
        pars_chain["tau_eff"] = np.zeros((chain.shape[0], zs.shape[0]))
        pars_chain["mF"] = np.zeros((chain.shape[0], zs.shape[0]))
        pars_chain["gamma"] = np.zeros((chain.shape[0], zs.shape[0]))
        pars_chain["sigT_kms"] = np.zeros((chain.shape[0], zs.shape[0]))
        pars_chain["T0"] = np.zeros((chain.shape[0], zs.shape[0]))

        pars_chain2 = {}
        pars_chain2["z"] = zs2
        # pars_chain2["tau_eff"] = np.zeros((chain.shape[0], zs2.shape[0]))
        pars_chain2["mF"] = np.zeros((chain.shape[0], zs2.shape[0]))
        pars_chain2["gamma"] = np.zeros((chain.shape[0], zs2.shape[0]))
        # pars_chain2["sigT_kms"] = np.zeros((chain.shape[0], zs2.shape[0]))
        pars_chain2["T0"] = np.zeros((chain.shape[0], zs2.shape[0]))

        for ii in range(chain.shape[0]):
            chain_params = parameter_space.values_from_cube(self.free_params, chain[ii, :])
            # pars_chain["tau_eff"][ii] = self.theory.model_igm.models[
            #     "F_model"
            # ].get_tau_eff(zs, like_params=chain_params)
            pars_chain["mF"][ii] = self.theory.model_igm.models[
                "F_model"
            ].get_mean_flux(zs, like_params=chain_params)
            pars_chain["gamma"][ii] = self.theory.model_igm.models[
                "T_model"
            ].get_gamma(zs, like_params=chain_params)
            # pars_chain["sigT_kms"][ii] = self.theory.model_igm.models[
            #     "T_model"
            # ].get_sigT_kms(zs, like_params=chain_params)
            pars_chain["T0"][ii] = (
                self.theory.model_igm.models["T_model"].get_T0(
                    zs, like_params=chain_params
                )
                / 1e4
            )

            pars_chain2["mF"][ii] = self.theory.model_igm.models[
                "F_model"
            ].get_mean_flux(zs2, like_params=chain_params)
            pars_chain2["gamma"][ii] = self.theory.model_igm.models[
                "T_model"
            ].get_gamma(zs2, like_params=chain_params)
            pars_chain2["T0"][ii] = (
                self.theory.model_igm.models["T_model"].get_T0(
                    zs2, like_params=chain_params
                )
                / 1e4
            )

        out["tab_out"].append(zs2)
        out["tab_out"].append(
            np.percentile(pars_chain2["mF"], [16, 50, 84], axis=0)
        )
        out["tab_out"].append(
            np.percentile(pars_chain2["T0"], [16, 50, 84], axis=0)
        )
        out["tab_out"].append(
            np.percentile(pars_chain2["gamma"], [16, 50, 84], axis=0)
        )
        # print(np.percentile(pars_chain2["mF"], [16, 50, 84], axis=0))
        # print(np.percentile(pars_chain["mF"], [16, 50, 84], axis=0))
        pars_chain2 = 0

    if free_params is not None:
        if zmask is not None:
            zs = zmask
        else:
            zs = primary_data.z
        pars_test = {}
        pars_test["z"] = zs
        pars_test["tau_eff"] = self.theory.model_igm.models["F_model"].get_tau_eff(
            zs, like_params=free_params
        )
        pars_test["mF"] = self.theory.model_igm.models["F_model"].get_mean_flux(
            zs, like_params=free_params
        )
        pars_test["gamma"] = self.theory.model_igm.models["T_model"].get_gamma(
            zs, like_params=free_params
        )
        pars_test["sigT_kms"] = self.theory.model_igm.models[
            "T_model"
        ].get_sigT_kms(zs, like_params=free_params)
        pars_test["T0"] = (
            self.theory.model_igm.models["T_model"].get_T0(
                zs, like_params=free_params
            )
            / 1e4
        )
        pars_test["kF_kms"] = self.theory.model_igm.models["P_model"].get_kF_kms(
            zs, like_params=free_params
        )

    # External IGM measurements are optional so synthetic-data plots can
    # show only the fit and its known truth.
    if plot_external_data:
        from cup1d.likelihood.likelihood import others_igm

        gal21, tu24 = others_igm()

    legend_ax = None
    empty_ax = None
    if plot_type == "all":
        fig, axes = plt.subplots(2, 3, figsize=(9, 6), sharex="col")
        # Reserve the right column for a readable legend rather than
        # covering the tau_eff history with it.
        ax = np.array([axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]])
        legend_ax = axes[0, 2]
        empty_ax = axes[1, 2]
        arr_labs = ["tau_eff", "gamma", "sigT_kms", "kF_kms"]
        latex_labs = [
            r"$\tau_\mathrm{eff}$",
            r"$\gamma$",
            r"$\sigma_\mathrm{T} [\mathrm{km\,s^{-1}}]$",
            r"$k_F$ [km/s]",
        ]
    elif plot_type == "tau_sigT":
        fig, ax = plt.subplots(
            3,
            1,
            figsize=(8, 10),
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1, 1]},
        )
        # arr_labs = ["tau_eff", "sigT_kms", "gamma"]
        # latex_labs = [
        #     r"$\tau_\mathrm{eff}$",
        #     r"$\sigma_\mathrm{T}\,\left[\mathrm{km\,s^{-1}}\right]$",
        #     r"$\gamma$",
        # ]
        arr_labs = ["mF", "T0", "gamma"]
        nexp_mF = 1
        latex_labs = [
            # r"$(1+z)\bar{F}$",
            r"$\bar{F}$",
            r"$T_0[K]/10^4$",
            r"$\gamma$",
        ]

    ax = np.asarray(ax).reshape(-1)

    for ii in range(len(arr_labs)):
        if plot_truth and self.truth is not None:
            truth_igm = self.truth["igm"]
            if arr_labs[ii] in truth_igm:
                truth_values = np.asarray(truth_igm[arr_labs[ii]])
                truth_z = np.asarray(truth_igm["z"])
                mask = truth_values != 0
                ax[ii].plot(
                    truth_z[mask],
                    truth_values[mask],
                    "C3:o",
                    alpha=0.8,
                    label="Truth",
                )

        if cloud:
            for jj, sim_label in enumerate(self.theory.emu_igm_all):
                if is_number_string(sim_label[-1]) == False:
                    continue
                if jj == 0:
                    lab = "Training data"
                    alpha = 0.75
                else:
                    lab = None
                    alpha = 0.1

                _ = np.argwhere(
                    self.theory.emu_igm_all[sim_label][arr_labs[ii]] != 0
                )[:, 0]
                if len(_) > 0:
                    ax[ii].scatter(
                        self.theory.emu_igm_all[sim_label]["z"][_],
                        self.theory.emu_igm_all[sim_label][arr_labs[ii]][_],
                        marker=".",
                        color="C1",
                        alpha=alpha,
                        label=lab,
                        s=10,
                    )

        if chain is not None:
            _ = pars_fid[arr_labs[ii]] != 0
            norm = 1
            ax[ii].fill_between(
                pars_fid["z"][_],
                norm * np.percentile(pars_chain[arr_labs[ii]][:, _], 5, axis=0),
                norm * np.percentile(pars_chain[arr_labs[ii]][:, _], 95, axis=0),
                color="lightblue",
                alpha=0.5,
            )
            ax[ii].fill_between(
                pars_fid["z"][_],
                norm * np.percentile(pars_chain[arr_labs[ii]][:, _], 16, axis=0),
                norm * np.percentile(pars_chain[arr_labs[ii]][:, _], 84, axis=0),
                color="C0",
                alpha=0.5,
            )

            if store_data:
                out["out_data"]["x" + str(ii) + "_blue_areas"] = pars_fid["z"][_]
                out["out_data"]["y" + str(ii) + "_blue_areas"] = np.percentile(
                    pars_chain[arr_labs[ii]][:, _], [5, 16, 84, 95], axis=0
                )

        if free_params is not None:
            _ = pars_test[arr_labs[ii]] != 0
            if arr_labs[ii] == "mF":
                # norm = (1 + pars_test["z"][_]) ** nexp_mF
                norm = 1
            else:
                norm = 1

            ax[ii].plot(
                pars_test["z"][_],
                norm * pars_test[arr_labs[ii]][_],
                "C0:",
                label=variation_label,
                alpha=1,
                lw=3,
            )

            if store_data:
                out["out_data"]["x" + str(ii) + "_blue_dotted"] = pars_test["z"][_]
                out["out_data"]["y" + str(ii) + "_blue_dotted"] = pars_test[
                    arr_labs[ii]
                ][_]

            if arr_labs[ii] == "mF":
                lab = "tau_eff_znodes"
            elif arr_labs[ii] == "T0":
                lab = "sigT_kms_znodes"
            else:
                lab = arr_labs[ii] + "_znodes"
            if lab in self.args.fid_igm:
                yy = np.interp(
                    self.args.fid_igm[lab],
                    pars_test["z"][_],
                    pars_test[arr_labs[ii]][_],
                )
                if arr_labs[ii] == "mF":
                    # norm = (1 + self.args.fid_igm[lab]) ** nexp_mF
                    norm = 1
                else:
                    norm = 1
                ax[ii].scatter(
                    self.args.fid_igm[lab],
                    norm * yy,
                    marker="o",
                    color="C0",
                )

                if store_data:
                    out["out_data"]["x" + str(ii) + "_blue_dots"] = (
                        self.args.fid_igm[lab]
                    )
                    out["out_data"]["y" + str(ii) + "_blue_dots"] = yy

        if plot_external_data and arr_labs[ii] == "tau_eff":
            ax[ii].errorbar(
                gal21["z"],
                gal21["tau_eff"],
                yerr=gal21["tau_eff_err"],
                fmt="--",
                color="C1",
                label="Gaikwad+2021",
                alpha=0.75,
                lw=2,
            )
            ax[ii].errorbar(
                tu24["z"],
                tu24["tau_eff"],
                yerr=tu24["tau_eff_err"],
                fmt="-.",
                color="C2",
                label="Turner+2024",
                alpha=0.75,
                lw=2,
            )
        elif plot_external_data and arr_labs[ii] == "sigT_kms":
            ax[ii].errorbar(
                gal21["z"],
                gal21["sigT_kms"],
                yerr=gal21["sigT_kms_err"],
                fmt="--",
                color="C1",
                label="Gaikwad+2021",
                alpha=0.75,
                lw=2,
            )
        elif plot_external_data and arr_labs[ii] == "mF":
            # norm = (1 + gal21["z"]) ** nexp_mF
            norm = 1
            ax[ii].errorbar(
                gal21["z"],
                norm * gal21["mF"],
                yerr=norm * gal21["mF_err"],
                fmt="--",
                color="C1",
                label="Gaikwad+2021",
                alpha=0.75,
                lw=2,
            )
            if store_data:
                out["out_data"]["x" + str(ii) + "_gal21"] = gal21["z"]
                out["out_data"]["y" + str(ii) + "_gal21"] = norm * gal21["mF"]
                out["out_data"]["yerr" + str(ii) + "_gal21"] = (
                    norm * gal21["mF_err"]
                )
            # norm = (1 + tu24["z"]) ** nexp_mF
            norm = 1
            ax[ii].errorbar(
                tu24["z"],
                norm * tu24["mF"],
                yerr=norm * tu24["mF_err"],
                fmt="-.",
                color="C2",
                label="Turner+2024",
                alpha=0.75,
                lw=2,
            )
            if store_data:
                out["out_data"]["x" + str(ii) + "_tur24"] = tu24["z"]
                out["out_data"]["y" + str(ii) + "_tur24"] = norm * tu24["mF"]
                out["out_data"]["yerr" + str(ii) + "_tur24"] = norm * tu24["mF_err"]
        elif plot_external_data and arr_labs[ii] == "T0":
            ax[ii].errorbar(
                gal21["z"],
                gal21["T0"],
                yerr=gal21["T0_err"],
                fmt="--",
                color="C1",
                label="Galdwick+2021",
                alpha=0.75,
                lw=2,
            )
            if store_data:
                out["out_data"]["x" + str(ii) + "_gal21"] = gal21["z"]
                out["out_data"]["y" + str(ii) + "_gal21"] = norm * gal21["T0"]
                out["out_data"]["yerr" + str(ii) + "_gal21"] = (
                    norm * gal21["T0_err"]
                )
        elif plot_external_data and arr_labs[ii] == "gamma":
            ax[ii].errorbar(
                gal21["z"],
                gal21["gamma"],
                yerr=gal21["gamma_err"],
                fmt="--",
                color="C1",
                label="Galdwick+2021",
                alpha=0.75,
                lw=2,
            )

            if store_data:
                out["out_data"]["x" + str(ii) + "_gal21"] = gal21["z"]
                out["out_data"]["y" + str(ii) + "_gal21"] = norm * gal21["gamma"]
                out["out_data"]["yerr" + str(ii) + "_gal21"] = (
                    norm * gal21["gamma_err"]
                )

    if plot_more_igm:
        more_igm_path = os.path.join(
            get_path_repo("cup1d"),
            "data",
            "tutorials",
            "data",
            "more_igm_data.npy",
        )
        more_igm = np.load(more_igm_path, allow_pickle=True).item()
        ax[0].plot(
            more_igm["z"],
            more_igm["mF"][1],
            "C5--",
            alpha=0.5,
            lw=3,
            label=r"IGM $n_z=6$",
        )
        ax[1].plot(more_igm["z"], more_igm["T0"][1], "C5--", alpha=0.5, lw=3)
        ax[2].plot(more_igm["z"], more_igm["gamma"][1], "C5--", alpha=0.5, lw=3)

        if store_data:
            out["out_data"]["x_brown"] = more_igm["z"]
            out["out_data"]["y0_brown"] = more_igm["mF"][1]
            out["out_data"]["y1_brown"] = more_igm["T0"][1]
            out["out_data"]["y2_brown"] = more_igm["gamma"][1]

    if plot_fid:
        for ii in range(len(arr_labs)):
            for kk in range(3):
                if kk == 0:
                    pars = pars_fid.copy()
                    label = lab_fid
                    lsk = "-"
                    alpha = 0.5
                    lw = 1.5
                elif kk == 1:
                    pars = pars_min.copy()
                    label = None
                    lsk = "-"
                    alpha = 0.3
                    lw = 1
                elif kk == 2:
                    pars = pars_max.copy()
                    label = None
                    lsk = "-"
                    alpha = 0.3
                    lw = 1

                _ = pars[arr_labs[ii]] != 0
                if arr_labs[ii] == "mF":
                    # norm = (1 + pars["z"][_]) ** nexp_mF
                    norm = 1
                else:
                    norm = 1

                ax[ii].plot(
                    pars["z"][_],
                    norm * pars[arr_labs[ii]][_],
                    "C3" + lsk,
                    label=label,
                    alpha=alpha,
                    lw=lw,
                )

                if store_data:
                    out["out_data"]["x" + str(ii) + "_red"] = pars["z"][_]
                    out["out_data"]["y" + str(ii) + "_red"] = (
                        norm * pars[arr_labs[ii]][_]
                    )

    for ii in range(len(arr_labs)):
        ax[ii].set_ylabel(latex_labs[ii], fontsize=ftsize)
        if ii == 0:
            if arr_labs[ii] == "tau_eff":
                ax[ii].set_yscale("log")
            if legend_ax is None:
                ax[ii].legend(fontsize=ftsize, loc="lower left", ncol=1)

        if (ii == 2) | (ii == len(arr_labs) - 1):
            ax[ii].set_xlabel(r"$z$", fontsize=ftsize)

        ax[ii].tick_params(axis="both", which="major", labelsize=ftsize)
        ax[ii].tick_params(axis="both", which="minor", labelsize=ftsize - 2)
        # A linear locator is inappropriate for the logarithmic
        # tau_eff panel, where it replaces Matplotlib's log ticks.
        if arr_labs[ii] != "tau_eff":
            ax[ii].yaxis.set_major_locator(MaxNLocator(nbins=3, prune=None))

    if legend_ax is not None:
        handles, labels = ax[0].get_legend_handles_labels()
        legend_ax.legend(handles, labels, fontsize=ftsize, loc="center")
        legend_ax.set_axis_off()
        empty_ax.set_axis_off()

    # These limits were designed for the three-panel mean-flux,
    # temperature, and gamma figure. Applying them to the four-panel
    # tau_eff/sigma_T view clips the plotted histories.
    if pre_xylims and plot_type == "tau_sigT":
        ax[0].set_ylim(0.35, 0.9)
        ax[1].set_ylim(0.0, 3.2)
        ax[2].set_ylim(0.8, 2.2)
    fig.suptitle(title, fontsize=ftsize + 2)
    plt.tight_layout()

    if save_directory is not None:
        name = os.path.join(save_directory, "IGM_histories")
        plt.savefig(name + ".pdf")
        plt.savefig(name + ".png")
    else:
        plt.show()

    return out


def plot_parameters(self, z, like_params, folder=None):
    """Plot likelihood parameters"""

    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(
        len(self.coeffs), 1, sharex=True, figsize=(8, 3 * len(self.coeffs))
    )
    if len(self.coeffs) == 1:
        ax = [ax]

    try:
        len_p = len(like_params[0])
    except:
        z_at_time = False
    else:
        z_at_time = True

    vals_out = {}
    coeffs_out = {}

    for ii, key in enumerate(self.coeffs.keys()):
        if z_at_time == False:
            if key == "tau_eff":
                vals = self.get_tau_eff(z, like_params=like_params)
            elif key == "gamma":
                vals = self.get_gamma(z, like_params=like_params)
            elif key == "sigT_kms":
                vals = self.get_sigT_kms(z, like_params=like_params)
            elif key == "kF_kms":
                vals = self.get_kF_kms(z, like_params=like_params)
            else:
                raise ValueError(
                    "key must be tau_eff, gamma, sigT_kms, or kF_kms"
                )
            coeffs_out[key] = self.get_coeff(key, like_params=like_params)
        else:
            vals = []
            coeffs_out[key] = []
            for jj in range(len(z)):
                if key == "tau_eff":
                    vals.append(
                        self.get_tau_eff(z[jj], like_params=like_params[jj])
                    )
                elif key == "gamma":
                    vals.append(
                        self.get_gamma(z[jj], like_params=like_params[jj])
                    )
                elif key == "sigT_kms":
                    vals.append(
                        self.get_sigT_kms(
                            z[jj], like_params=like_params[jj]
                        )
                    )
                elif key == "kF_kms":
                    vals.append(
                        self.get_kF_kms(z[jj], like_params=like_params[jj])
                    )
                else:
                    raise ValueError(
                        "key must be tau_eff, gamma, sigT_kms, or kF_kms"
                    )
                coeffs_out[key].append(
                    self.get_coeff(key, like_params=like_params[jj])[0]
                )
            vals = np.array(vals)

        if key == "tau_eff":
            fid_vals = self.get_tau_eff(z)
        elif key == "gamma":
            fid_vals = self.get_gamma(z)
        elif key == "sigT_kms":
            fid_vals = self.get_sigT_kms(z)
        elif key == "kF_kms":
            fid_vals = self.get_kF_kms(z)

        if self.prop_coeffs[key + "_otype"] == "exp":
            vals = np.log(vals)
            fid_vals = np.log(fid_vals)

        vals_out[key] = vals

        ax[ii].plot(z, vals, "o-", label="data")
        res = np.polyfit(z, vals, 1)
        ax[ii].plot(z, res[0] * z + res[1], "--", label="fit")
        ax[ii].plot(z, fid_vals, "-.", label="fid")
        ax[ii].set_ylabel(key)
    ax[0].legend()
    ax[-1].set_xlabel("z")

    plt.tight_layout()
    plt.show()

    if folder is not None:
        fig.savefig(folder + ".png")
        fig.savefig(folder + ".pdf")

    return vals_out, coeffs_out


def plot_mock_igm(self):
    """Plot IGM histories"""

    # true IGM parameters
    pars_true = {}
    pars_true["z"] = self.truth["igm"]["z"]
    pars_true["tau_eff"] = self.truth["igm"]["tau_eff"]
    pars_true["gamma"] = self.truth["igm"]["gamma"]
    pars_true["sigT_kms"] = self.truth["igm"]["sigT_kms"]
    pars_true["kF_kms"] = self.truth["igm"]["kF_kms"]

    fig, ax = plt.subplots(2, 2, figsize=(6, 6), sharex=True)
    ax = ax.reshape(-1)

    arr_labs = ["tau_eff", "gamma", "sigT_kms", "kF_kms"]
    latex_labs = [
        r"$\tau_\mathrm{eff}$",
        r"$\gamma$",
        r"$\sigma_T$",
        r"$k_F$",
    ]

    for ii in range(len(arr_labs)):
        _ = pars_true[arr_labs[ii]] != 0
        ax[ii].plot(
            pars_true["z"][_],
            pars_true[arr_labs[ii]][_],
            "o:",
            label="true",
        )

        ax[ii].set_ylabel(latex_labs[ii])
        if ii == 0:
            ax[ii].set_yscale("log")

        if (ii == 2) | (ii == 3):
            ax[ii].set_xlabel(r"$z$")

    plt.tight_layout()
