"""Contaminants plotting implementations; scientific objects retain compatibility wrappers."""

import numpy as np

from cup1d.likelihood import parameter as parameter_space
import matplotlib.pyplot as plt
import os
from cup1d.postprocessing.style import get_discrete_cmap


def plot_hcd_cont(
    self,
    zstar=3,
    p0=None,
    chain=None,
    save_directory=None,
    ftsize=24,
    nelem=5000,
    store_data=False,
):
    if store_data:
        out_data = {}

    if chain is not None:
        if len(chain.shape) == 3:
            chain_use = chain.reshape(-1, chain.shape[-1])
        else:
            chain_use = chain.copy()
        ind = np.random.permutation(np.arange(0, chain_use.shape[0]))[:nelem]
        chain_use = chain_use[ind]

    ind = np.argwhere(self.data.z == zstar)[0][0]
    k_kms_inter = np.linspace(
        self.data.k_kms[ind].min(), self.data.k_kms[ind].max(), 500
    )

    labels = ["LLS", "sub-DLA", "small DLA", "large DLA", "All"]
    fig, ax = plt.subplots(figsize=(8, 6))
    ls = ["--", "-.", (0, (2, 2, 2, 2)), ":", "-"]

    for ii in range(5):
        par_plot = "HCD_damp" + str(ii + 1)
        # print(par_plot)

        if chain is None:
            free_params = parameter_space.values_from_cube(self.free_params, p0)
            for name in free_params:
                if ii + 1 <= 4 and "HCD_damp" in name:
                    if not name.startswith(par_plot):
                        free_params[name] = -20

            hcd_cont = self.theory.model_cont.hcd_model.get_contamination(
                z=np.array([zstar]),
                k_kms=[k_kms_inter],
                like_params=free_params,
            )
            ax.plot(
                k_kms_inter,
                hcd_cont,
                label=labels[ii],
                alpha=0.75,
                ls=ls[ii],
                lw=3,
                color="C" + str(ii),
            )
        else:
            all_hcd_cont = np.zeros((nelem, len(k_kms_inter)))

            for jj in range(nelem):
                free_params = parameter_space.values_from_cube(self.free_params, chain_use[jj])
                for name in free_params:
                    if ii + 1 <= 4 and "HCD_damp" in name:
                        if not name.startswith(par_plot):
                            free_params[name] = -20
                all_hcd_cont[jj, :] = (
                    self.theory.model_cont.hcd_model.get_contamination(
                        z=np.array([zstar]),
                        k_kms=[k_kms_inter],
                        like_params=free_params,
                    )
                )
            hcd_cont = np.percentile(all_hcd_cont, [16, 50, 84], axis=0)

            if store_data:
                out_data["x"] = k_kms_inter
                out_data["y" + str(ii)] = hcd_cont

            ax.plot(
                k_kms_inter,
                hcd_cont[1],
                label=labels[ii],
                alpha=0.75,
                ls=ls[ii],
                lw=3,
                color="C" + str(ii),
            )
            ax.fill_between(
                k_kms_inter,
                hcd_cont[0],
                hcd_cont[2],
                alpha=0.3,
                color="C" + str(ii),
            )
    ax.axhline(1, color="k", ls=":", lw=2)
    ax.set_ylabel(r"$C_\mathrm{HCD}$", fontsize=ftsize)
    ax.set_xlabel(r"$k_\parallel\, [\mathrm{km}^{-1}\mathrm{s}]$", fontsize=ftsize)
    ax.tick_params(axis="both", which="major", labelsize=ftsize)
    ax.legend(fontsize=ftsize - 2)

    plt.tight_layout()

    if save_directory is not None:
        name = os.path.join(save_directory, "cont_hcd")
        plt.savefig(name + ".pdf")
        plt.savefig(name + ".png")
    else:
        plt.show()

    if store_data:
        return out_data


def plot_metal_cont_add(
    self,
    free_params=None,
    chain=None,
    save_directory=None,
    ftsize=24,
    nelem=5000,
    store_data=False,
):
    if store_data:
        out_data = {}

    if chain is not None:
        if len(chain.shape) == 3:
            chain_use = chain.reshape(-1, chain.shape[-1])
        else:
            chain_use = chain.copy()
        ind = np.random.permutation(np.arange(0, chain_use.shape[0]))[:nelem]
        chain_use = chain_use[ind]

    fig, ax = plt.subplots(figsize=(8, 6))
    ls = ["-", "--", "-.", ":"]
    for ii, zstar in enumerate([2.2, 2.8, 3.4, 4.0]):
        ind = np.argwhere(self.data.z == zstar)[0][0]
        k_kms_inter = np.linspace(
            self.data.k_kms[ind].min(), self.data.k_kms[ind].max(), 500
        )
        k_kms = self.data.k_kms[ind].copy()
        mF = self.theory.model_igm.models["F_model"].get_mean_flux(
            zstar, like_params=free_params
        )

        if chain is None:
            si_add_cont_all = self.theory.model_cont.metal_models[
                "Si_add"
            ].get_contamination(
                z=np.array([zstar]),
                k_kms=[k_kms_inter],
                mF=np.array([mF]),
                like_params=free_params,
            )
            ax.plot(
                k_kms_inter,
                si_add_cont_all[0],
                label=r"$z=$" + str(zstar),
                alpha=0.75,
                ls=ls[ii],
                lw=4,
                color="C" + str(ii),
            )
        else:
            all_si_add_cont = np.zeros((nelem, len(k_kms_inter)))
            for jj in range(nelem):
                free_params = parameter_space.values_from_cube(self.free_params, chain_use[jj])
                all_si_add_cont[jj, :] = self.theory.model_cont.metal_models[
                    "Si_add"
                ].get_contamination(
                    z=np.array([zstar]),
                    k_kms=[k_kms_inter],
                    mF=np.array([mF]),
                    like_params=free_params,
                )[
                    0
                ]
            si_add_cont = np.percentile(all_si_add_cont, [16, 50, 84], axis=0)

            if store_data:
                out_data["x"] = k_kms_inter
                out_data["y" + str(ii)] = si_add_cont

            ax.plot(
                k_kms_inter,
                si_add_cont[1],
                label=r"$z=$" + str(zstar),
                alpha=0.75,
                ls=ls[ii],
                lw=2,
                color="C" + str(ii),
            )
            ax.fill_between(
                k_kms_inter,
                si_add_cont[0],
                si_add_cont[2],
                alpha=0.3,
                color="C" + str(ii),
            )

    ax.axhline(0, color="k", ls=":", lw=2)
    ax.legend(fontsize=ftsize - 4, loc="upper right")
    ax.tick_params(axis="both", which="major", labelsize=ftsize)
    ax.set_ylabel(
        r"$C_\mathrm{SiII-SiII}\,[\mathrm{km}\,\mathrm{s}^{-1}]$",
        fontsize=ftsize,
    )
    ax.set_xlabel(r"$k_\parallel\,[\mathrm{km}^{-1} \mathrm{s}]$", fontsize=ftsize)
    ax.set_ylim(-0.2, 3)

    plt.tight_layout()

    if save_directory is not None:
        name = os.path.join(save_directory, "cont_metal_add")
        plt.savefig(name + ".pdf", bbox_inches="tight")
        plt.savefig(name + ".png", bbox_inches="tight")
    else:
        plt.show()

    if store_data:
        return out_data


def plot_metal_cont_mult(
    self,
    free_params=None,
    chain=None,
    zstar=3,
    save_directory=None,
    ftsize=24,
    nelem=5000,
    store_data=False,
):
    """Plot metallicity contours"""

    if store_data:
        out_data = {}

    if chain is not None:
        if len(chain.shape) == 3:
            chain_use = chain.reshape(-1, chain.shape[-1])
        else:
            chain_use = chain.copy()
        ind = np.random.permutation(np.arange(0, chain_use.shape[0]))[:nelem]
        chain_use = chain_use[ind]

    ind = np.argwhere(self.data.z == zstar)[0][0]
    k_kms_inter = np.linspace(
        self.data.k_kms[ind].min(), self.data.k_kms[ind].max(), 500
    )
    # k_kms = self.data.k_kms[ind].copy()

    # dat_si_mult_cont_all = self.theory.model_cont.metal_models[
    #     "Si_mult"
    # ].get_contamination(
    #     z=np.array([zstar]),
    #     k_kms=[k_kms],
    #     mF=np.array([mF]),
    #     like_params=free_params,
    # )

    if chain is None:
        mF = self.theory.model_igm.models["F_model"].get_mean_flux(
            zstar, like_params=free_params
        )

        si_mult_cont_all = self.theory.model_cont.metal_models[
            "Si_mult"
        ].get_contamination(
            z=np.array([zstar]),
            k_kms=[k_kms_inter],
            mF=np.array([mF]),
            like_params=free_params,
        )

        remove = {
            "SiIII_Lya": 1,
            "SiIIa_Lya": 0,
            "SiIIb_Lya": 0,
            "SiIIc_Lya": 0,
            "SiIII_SiIIa": 0,
            "SiIII_SiIIb": 0,
            "SiIII_SiIIc": 0,
            "SiIIc_SiIIb": 0,
            "SiIIc_SiIIa": 0,
            "SiIIb_SiIIa": 0,
        }

        si_mult_cont_SiIII = self.theory.model_cont.metal_models[
            "Si_mult"
        ].get_contamination(
            z=np.array([zstar]),
            k_kms=[k_kms_inter],
            mF=np.array([mF]),
            like_params=free_params,
            remove=remove,
        )

        remove = {
            "SiIII_Lya": 0,
            "SiIIa_Lya": 1,
            "SiIIb_Lya": 1,
            "SiIIc_Lya": 0,
            "SiIII_SiIIa": 0,
            "SiIII_SiIIb": 0,
            "SiIII_SiIIc": 0,
            "SiIIc_SiIIb": 0,
            "SiIIc_SiIIa": 0,
            "SiIIb_SiIIa": 0,
        }

        si_mult_cont_SiII = self.theory.model_cont.metal_models[
            "Si_mult"
        ].get_contamination(
            z=np.array([zstar]),
            k_kms=[k_kms_inter],
            mF=np.array([mF]),
            like_params=free_params,
            remove=remove,
        )

        remove = {
            "SiIII_Lya": 0,
            "SiIIa_Lya": 0,
            "SiIIb_Lya": 0,
            "SiIIc_Lya": 0,
            "SiIII_SiIIa": 1,
            "SiIII_SiIIb": 1,
            "SiIII_SiIIc": 0,
            "SiIIc_SiIIb": 0,
            "SiIIc_SiIIa": 0,
            "SiIIb_SiIIa": 0,
        }

        si_mult_cont_Si23 = self.theory.model_cont.metal_models[
            "Si_mult"
        ].get_contamination(
            z=np.array([zstar]),
            k_kms=[k_kms_inter],
            mF=np.array([mF]),
            like_params=free_params,
            remove=remove,
        )

        fig, ax = plt.subplots(4, figsize=(8, 6), sharey=True, sharex=True)
        ax[0].plot(
            k_kms_inter,
            si_mult_cont_SiIII[0],
            label=r"Ly$\alpha$-SiIII",
            alpha=0.75,
            ls="-",
            lw=3,
            color="C0",
        )
        ax[1].plot(
            k_kms_inter,
            si_mult_cont_SiII[0],
            label=r"Ly$\alpha$-SiII",
            alpha=0.75,
            ls="-",
            lw=3,
            color="C1",
        )
        ax[2].plot(
            k_kms_inter,
            si_mult_cont_Si23[0],
            label=r"SiII-SiIII",
            alpha=0.75,
            ls="-",
            lw=3,
            color="C2",
        )
        ax[3].plot(
            k_kms_inter,
            si_mult_cont_all[0],
            label=r"All",
            alpha=0.75,
            ls="-",
            lw=3,
            color="C3",
        )
    else:
        si_mult_cont_all = np.zeros((nelem, len(k_kms_inter)))
        si_mult_cont_SiIII = np.zeros((nelem, len(k_kms_inter)))
        si_mult_cont_SiII = np.zeros((nelem, len(k_kms_inter)))
        si_mult_cont_Si23 = np.zeros((nelem, len(k_kms_inter)))

        for jj in range(nelem):
            free_params = parameter_space.values_from_cube(self.free_params, chain_use[jj])

            mF = self.theory.model_igm.models["F_model"].get_mean_flux(
                zstar, like_params=free_params
            )

            si_mult_cont_all[jj] = self.theory.model_cont.metal_models[
                "Si_mult"
            ].get_contamination(
                z=np.array([zstar]),
                k_kms=[k_kms_inter],
                mF=np.array([mF]),
                like_params=free_params,
            )[
                0
            ]

            remove = {
                "SiIII_Lya": 1,
                "SiIIa_Lya": 0,
                "SiIIb_Lya": 0,
                "SiIIc_Lya": 0,
                "SiIII_SiIIa": 0,
                "SiIII_SiIIb": 0,
                "SiIII_SiIIc": 0,
                "SiIIc_SiIIb": 0,
                "SiIIc_SiIIa": 0,
                "SiIIb_SiIIa": 0,
            }

            si_mult_cont_SiIII[jj] = self.theory.model_cont.metal_models[
                "Si_mult"
            ].get_contamination(
                z=np.array([zstar]),
                k_kms=[k_kms_inter],
                mF=np.array([mF]),
                like_params=free_params,
                remove=remove,
            )[
                0
            ]

            remove = {
                "SiIII_Lya": 0,
                "SiIIa_Lya": 1,
                "SiIIb_Lya": 1,
                "SiIIc_Lya": 0,
                "SiIII_SiIIa": 0,
                "SiIII_SiIIb": 0,
                "SiIII_SiIIc": 0,
                "SiIIc_SiIIb": 0,
                "SiIIc_SiIIa": 0,
                "SiIIb_SiIIa": 0,
            }

            si_mult_cont_SiII[jj] = self.theory.model_cont.metal_models[
                "Si_mult"
            ].get_contamination(
                z=np.array([zstar]),
                k_kms=[k_kms_inter],
                mF=np.array([mF]),
                like_params=free_params,
                remove=remove,
            )[
                0
            ]

            remove = {
                "SiIII_Lya": 0,
                "SiIIa_Lya": 0,
                "SiIIb_Lya": 0,
                "SiIIc_Lya": 0,
                "SiIII_SiIIa": 1,
                "SiIII_SiIIb": 1,
                "SiIII_SiIIc": 0,
                "SiIIc_SiIIb": 0,
                "SiIIc_SiIIa": 0,
                "SiIIb_SiIIa": 0,
            }

            si_mult_cont_Si23[jj] = self.theory.model_cont.metal_models[
                "Si_mult"
            ].get_contamination(
                z=np.array([zstar]),
                k_kms=[k_kms_inter],
                mF=np.array([mF]),
                like_params=free_params,
                remove=remove,
            )[
                0
            ]

        per_siIII = np.percentile(si_mult_cont_SiIII, [16, 50, 84], axis=0)
        per_siII = np.percentile(si_mult_cont_SiII, [16, 50, 84], axis=0)
        per_si23 = np.percentile(si_mult_cont_Si23, [16, 50, 84], axis=0)
        per_siall = np.percentile(si_mult_cont_all, [16, 50, 84], axis=0)

        if store_data:
            out_data["x"] = k_kms_inter
            out_data["y_blue"] = per_siIII
            out_data["y_orange"] = per_siII
            out_data["y_green"] = per_si23
            out_data["y_red"] = per_siall

        fig, ax = plt.subplots(4, figsize=(8, 6), sharey=True, sharex=True)
        ax[0].plot(
            k_kms_inter,
            per_siIII[1],
            label=r"Ly$\alpha$-SiIII",
            alpha=0.75,
            ls="-",
            lw=2,
            color="C0",
        )
        ax[0].fill_between(
            k_kms_inter,
            per_siIII[0],
            per_siIII[2],
            alpha=0.3,
            color="C0",
        )

        ax[1].plot(
            k_kms_inter,
            per_siII[1],
            label=r"Ly$\alpha$-SiII",
            alpha=0.75,
            ls="-",
            lw=2,
            color="C1",
        )
        ax[1].fill_between(
            k_kms_inter,
            per_siII[0],
            per_siII[2],
            alpha=0.3,
            color="C1",
        )

        ax[2].plot(
            k_kms_inter,
            per_si23[1],
            label=r"SiII-SiIII",
            alpha=0.75,
            ls="-",
            lw=2,
            color="C2",
        )
        ax[2].fill_between(
            k_kms_inter,
            per_si23[0],
            per_si23[2],
            alpha=0.3,
            color="C2",
        )

        ax[3].plot(
            k_kms_inter,
            per_siall[1],
            label=r"All",
            alpha=0.75,
            ls="-",
            lw=2,
            color="C3",
        )
        ax[3].fill_between(
            k_kms_inter,
            per_siall[0],
            per_siall[2],
            alpha=0.3,
            color="C3",
        )

    # ax[3].scatter(k_kms, dat_si_mult_cont_all[0], s=30, color="C3")
    for ii in range(4):
        ax[ii].axhline(1, color="k", ls=":", lw=2)
        ax[ii].legend(fontsize=ftsize - 4, loc="lower right")
        ax[ii].tick_params(axis="both", which="major", labelsize=ftsize)
    fig.supylabel(r"$C_\mathrm{metal}$", fontsize=ftsize)
    ax[-1].set_xlabel(
        r"$k_\parallel\,[\mathrm{km}^{-1} \mathrm{s}]$", fontsize=ftsize
    )

    plt.tight_layout()

    if save_directory is not None:
        name = os.path.join(save_directory, "cont_metal_mult")
        plt.savefig(name + ".pdf")
        plt.savefig(name + ".png")
    else:
        plt.show()

    if store_data:
        return out_data


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
            vals = self.get_value(key, z, like_params=like_params)
            coeffs_out[key] = self.get_coeff(key, like_params=like_params)
        else:
            vals = []
            coeffs_out[key] = []
            for jj in range(len(z)):
                vals.append(
                    self.get_value(key, z[jj], like_params=like_params[jj])
                )
                coeffs_out[key].append(
                    self.get_coeff(key, like_params=like_params[jj])[0]
                )
            vals = np.array(vals)

        if key in self.null_vals:
            if np.all(vals == self.null_vals[key]):
                continue
        elif key == "HCD_const":
            if np.all(vals == 0):
                continue

        if self.prop_coeffs[key + "_otype"] == "exp":
            vals = np.log(vals)

        vals_out[key] = vals

        _ = vals != self.null_vals[key]
        ax[ii].plot(z[_], vals[_], "o-", label="data")
        xz = np.log((1 + z) / (1 + self.z_0))
        if np.any(_):
            res = np.polyfit(xz[_], vals[_], 1)
            ax[ii].plot(z[_], res[0] * xz[_] + res[1], "--", label="fit")
            ax[ii].set_ylabel(key)

    ax[0].legend()
    ax[-1].set_xlabel("z")

    plt.tight_layout()
    plt.show()
    if folder is not None:
        fig.savefig(folder + ".png")
        fig.savefig(folder + ".pdf")

    return vals_out, coeffs_out


def plot_hcd_contamination(
    self,
    z,
    k_kms,
    ln_A_damp_coeff=None,
    plot_every_iz=1,
    cmap=None,
    smooth_k=False,
):
    """Plot the contamination model"""

    from cup1d.models.contaminants.HCD.hcd_model_McDonald2005 import HCD_Model_McDonald2005

    from matplotlib import pyplot as plt

    # plot for fiducial value
    if ln_A_damp_coeff is None:
        ln_A_damp_coeff = self.ln_A_damp_coeff

    hcd_model = HCD_Model_McDonald2005(ln_A_damp_coeff=ln_A_damp_coeff)

    for ii in range(0, len(z), plot_every_iz):
        if smooth_k:
            k_use = np.logspace(
                np.log10(k_kms[ii][0]), np.log10(k_kms[ii][-1]), 200
            )
        else:
            k_use = k_kms[ii]

        cont = hcd_model.get_contamination(z[ii], k_use)
        if isinstance(cont, int):
            cont = np.ones_like(k_use)
        if cmap is None:
            plt.plot(k_use, cont, label="z=" + str(z[ii]))
        else:
            plt.plot(k_use, cont, color=cmap(ii), label="z=" + str(z[ii]))

    plt.axhline(1, color="k", linestyle=":")

    plt.legend()
    plt.xscale("log")
    plt.xlabel(r"$k$ [1/Mpc]")
    plt.ylabel("HCD contamination")
    plt.tight_layout()

    return


def plot_agn_contamination(
    self,
    z,
    k_kms,
    ln_AGN_coeff=None,
    plot_every_iz=1,
    cmap=None,
    smooth_k=False,
    dict_data=None,
    zrange=[0, 10],
    name=None,
):
    """Plot the contamination model"""

    from cup1d.models.contaminants.feedback.AGN_model import AGN_Model

    # plot for fiducial value
    if ln_AGN_coeff is None:
        ln_AGN_coeff = self.ln_AGN_coeff

    if cmap is None:
        cmap = get_discrete_cmap(len(z))

    agn_model = AGN_Model(ln_AGN_coeff=ln_AGN_coeff)

    yrange = [1, 1]
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    fig2, ax2 = plt.subplots(
        len(z), sharex=True, sharey=True, figsize=(8, len(z) * 4)
    )
    if len(z) == 1:
        ax2 = [ax2]

    for ii in range(0, len(z), plot_every_iz):
        if dict_data is not None:
            indz = np.argwhere(np.abs(dict_data["zs"] - z[ii]) < 1.0e-3)[
                :, 0
            ]
            if len(indz) != 1:
                continue
            else:
                indz = indz[0]

        if (z[ii] > zrange[1]) | (z[ii] < zrange[0]):
            continue

        if smooth_k:
            k_use = np.logspace(
                np.log10(k_kms[ii][0]), np.log10(k_kms[ii][-1]), 200
            )
        else:
            k_use = k_kms[ii]
        cont = agn_model.get_contamination(z[ii], k_use)
        if isinstance(cont, int):
            cont = np.ones_like(k_use)

        ax1.plot(k_use, cont, color=cmap(ii), label="z=" + str(z[ii]))
        ax2[ii].plot(k_use, cont, color=cmap(ii), label="z=" + str(z[ii]))

        yrange[0] = min(yrange[0], np.min(cont))
        yrange[1] = max(yrange[1], np.max(cont))

        if dict_data is not None:
            yy = (
                dict_data["p1d_data"][indz]
                / dict_data["p1d_model"][indz]
                * cont
            )
            err_yy = (
                dict_data["p1d_err"][indz]
                / dict_data["p1d_model"][indz]
                * cont
            )

            ax1.errorbar(
                dict_data["k_kms"][indz],
                yy,
                err_yy,
                marker="o",
                linestyle=":",
                color=cmap(ii),
                alpha=0.5,
            )
            ax2[ii].errorbar(
                dict_data["k_kms"][indz],
                yy,
                err_yy,
                marker="o",
                linestyle=":",
                color=cmap(ii),
                alpha=0.5,
            )

    ax1.axhline(1, color="k", linestyle=":")
    ax1.legend(ncol=4)
    ax1.set_ylim(yrange[0] * 0.95, yrange[1] * 1.05)
    ax1.set_xscale("log")
    ax1.set_xlabel(r"$k$ [1/Mpc]")
    ax1.set_ylabel(r"$P_\mathrm{1D}/P_\mathrm{1D}^\mathrm{no\,AGN}$")
    for ax in ax2:
        ax.axhline(1, color="k", linestyle=":")
        ax.legend()
        ax.set_ylim(yrange[0] * 0.95, yrange[1] * 1.05)
        ax.set_xlabel(r"$k$ [1/Mpc]")
        ax.set_ylabel(r"$P_\mathrm{1D}/P_\mathrm{1D}^\mathrm{no\,AGN}$")
        ax.set_xscale("log")

    fig1.tight_layout()
    fig2.tight_layout()

    if name is None:
        fig1.show()
        fig2.show()
    else:
        if len(z) != 1:
            fig1.savefig(name + "_all.pdf")
            fig1.savefig(name + "_all.png")
        fig2.savefig(name + "_z.pdf")
        fig2.savefig(name + "_z.png")

    return
