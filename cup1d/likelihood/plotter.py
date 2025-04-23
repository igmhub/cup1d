import inspect
import matplotlib.pyplot as plt
from corner import corner
import numpy as np
import os
from cup1d.utils.utils import get_discrete_cmap, get_path_repo, purge_chains


class Plotter(object):
    def __init__(
        self, fitter=None, save_directory=None, fname_chain=None, args={}
    ):
        if fitter is not None:
            self.fitter = fitter
        elif fname_chain is not None:
            from cup1d.likelihood.input_pipeline import Args
            from cup1d.likelihood.pipeline import Pipeline

            # load file with chain
            data = np.load(fname_chain, allow_pickle=True).item()

            # set input args to pipeline and evaluate
            args_possible = inspect.signature(Args)
            dict_input = {}
            for param in args_possible.parameters.values():
                if param.name in data["args"].keys():
                    if param.name in args:
                        dict_input[param.name] = args[param.name]
                    else:
                        dict_input[param.name] = data["args"][param.name]
                    # print(param.name, data["args"][param.name])
                else:
                    print(param.name)

            args = Args(**dict_input)
            # args.p1d_fname = "/home/jchaves/Proyectos/projects/lya/data/cup1d/obs/p1d_fft_y1_measurement_kms_v6.fits"
            self.fitter = Pipeline(args, out_folder=save_directory).fitter

            # add sampler results to fitter
            self.fitter.mle_cube = data["fitter"]["mle_cube"]
            self.fitter.mle_cosmo = data["fitter"]["mle_cosmo"]
            self.fitter.mle = data["fitter"]["mle"]
            self.fitter.lnprop_mle = data["fitter"]["lnprob_mle"]
            if "lnprob" in data["fitter"].keys():
                self.fitter.lnprob = data["fitter"]["lnprob"]
                self.fitter.chain = data["fitter"]["chain"]
                self.fitter.blobs = data["fitter"]["blobs"]
        else:
            ValueError("Provide either fitter or fname_chain")

        self.cmap = get_discrete_cmap(len(self.fitter.like.data.z))
        self.save_directory = save_directory
        if save_directory is not None:
            os.makedirs(save_directory, exist_ok=True)

        self.mle_values = self.fitter.get_best_fit(stat_best_fit="mle")
        self.like_params = self.fitter.like.parameters_from_sampling_point(
            self.mle_values
        )
        self.mle_results = self.fitter.like.plot_p1d(
            values=self.mle_values,
            plot_every_iz=1,
            return_all=True,
            show=False,
        )

    def plots_minimizer(self, zrange=[0, 10]):
        plt.close("all")
        # plot initial P1D (before fitting)
        self.plot_P1D_initial(residuals=False)
        plt.close()
        self.plot_P1D_initial(residuals=True)
        plt.close()

        # plot best fit
        self.plot_p1d(residuals=False, stat_best_fit="mle")
        plt.close()
        self.plot_p1d(residuals=True, stat_best_fit="mle")
        plt.close()

        # plot cosmology
        if self.fitter.fix_cosmology == False:
            self.plot_mle_cosmo()
            plt.close()

        # plot IGM histories
        self.plot_igm(cloud=True)
        plt.close()

        # plot contamination
        self.plot_hcd_cont(plot_data=True, zrange=zrange)
        plt.close()
        self.plot_metal_cont(smooth_k=True, plot_data=True, zrange=zrange)
        plt.close()
        self.plot_agn_cont(plot_data=True, zrange=zrange)
        plt.close()

        # plot fit in hull
        self.plot_hull()
        plt.close()

    def plots_sampler(self):
        # plot lnprob
        plt.close("all")
        self.plot_lnprob()
        plt.close()

        # plot initial P1D (before fitting)
        self.plot_P1D_initial(residuals=False)
        plt.close()
        self.plot_P1D_initial(residuals=True)
        plt.close()

        # plot best fit
        self.plot_p1d(residuals=False, stat_best_fit="mle")
        plt.close()
        self.plot_p1d(residuals=True, stat_best_fit="mle")
        plt.close()

        # plot cosmology
        if self.fitter.fix_cosmology == False:
            self.plot_corner(only_cosmo=True, only_cosmo_lims=False)
            plt.close()
            self.plot_corner(only_cosmo=True, only_cosmo_lims=True)
            plt.close()

        # plot corner
        self.plot_corner()
        plt.close()

        # plot IGM histories
        self.plot_igm(cloud=True)
        plt.close()

        # plot contamination
        self.plot_hcd_cont()
        plt.close()
        self.plot_metal_cont()
        # plt.close()
        self.plot_agn_cont()
        plt.close()

        # plot fit in hull
        self.plot_hull()
        plt.close()

    def get_hc_star(self, nyx_version="Jul2024"):
        suite_emu = self.fitter.like.theory.emulator.list_sim_cube[0][:3]
        if suite_emu == "mpg":
            fname = os.path.join(
                get_path_repo("lace"),
                "data",
                "sim_suites",
                "Australia20",
                "mpg_emu_cosmo.npy",
            )
        elif suite_emu == "nyx":
            fname = os.path.join(
                os.environ["NYX_PATH"], "nyx_emu_cosmo_" + nyx_version + ".npy"
            )
        else:
            ValueError("cosmo_label should be 'mpg' or 'nyx'")

        try:
            data_cosmo = np.load(fname, allow_pickle=True).item()
        except:
            ValueError(f"{fname} not found")

        labs = []
        delta2_star = np.zeros(len(data_cosmo))
        n_star = np.zeros(len(data_cosmo))
        alpha_star = np.zeros(len(data_cosmo))

        for ii, key in enumerate(data_cosmo):
            labs.append(key)
            delta2_star[ii] = data_cosmo[key]["star_params"]["Delta2_star"]
            n_star[ii] = data_cosmo[key]["star_params"]["n_star"]
            alpha_star[ii] = data_cosmo[key]["star_params"]["alpha_star"]
        return labs, delta2_star, n_star, alpha_star, suite_emu, data_cosmo

    def plot_mle_cosmo(self, fontsize=16):
        """Plot MLE cosmology"""

        (
            labs,
            delta2_star,
            n_star,
            alpha_star,
            suite_emu,
            data_cosmo,
        ) = self.get_hc_star()

        if suite_emu == "mpg":
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax = [ax]
        else:
            fig, ax = plt.subplots(1, 3, figsize=(14, 6))

        ax[0].scatter(delta2_star, n_star)
        if suite_emu != "mpg":
            ax[1].scatter(delta2_star, alpha_star)
            ax[2].scatter(n_star, alpha_star)

        dif0 = delta2_star.max() - delta2_star.min()
        dif1 = n_star.max() - n_star.min()
        dif2 = alpha_star.max() - alpha_star.min()
        sep_x = 0.01
        for ii, lab in enumerate(labs):
            if data_cosmo[lab]["sim_label"][-1].isdigit():
                sep_y = 0
            else:
                sep_y = 0.01
            ax[0].annotate(
                lab[4:],
                (delta2_star[ii] + sep_x * dif0, n_star[ii] + sep_y * dif1),
                fontsize=8,
            )
            if suite_emu != "mpg":
                ax[1].annotate(
                    lab[4:],
                    (
                        delta2_star[ii] + sep_x * dif0,
                        alpha_star[ii] + sep_y * dif2,
                    ),
                    fontsize=8,
                )
                ax[2].annotate(
                    lab[4:],
                    (n_star[ii] + sep_x * dif1, alpha_star[ii] + sep_y * dif2),
                    fontsize=8,
                )

        for ii in range(2):
            # best
            if ii == 0:
                marker = "X"
                color = "C1"
                Delta2_star = self.fitter.mle_cosmo["Delta2_star"]
                n_star = self.fitter.mle_cosmo["n_star"]
                if suite_emu != "mpg":
                    alpha_star = self.fitter.mle_cosmo["alpha_star"]
                label = "MLE"
            # truth
            else:
                if self.fitter.like.truth is not None:
                    marker = "s"
                    color = "C3"
                    Delta2_star = self.fitter.like.truth["linP"]["Delta2_star"]
                    n_star = self.fitter.like.truth["linP"]["n_star"]
                    if suite_emu != "mpg":
                        alpha_star = self.fitter.like.truth["linP"][
                            "alpha_star"
                        ]
                    label = "truth"
                else:
                    continue

            ax[0].scatter(
                Delta2_star, n_star, marker=marker, color=color, label=label
            )
            if suite_emu != "mpg":
                ax[1].scatter(
                    Delta2_star,
                    alpha_star,
                    marker=marker,
                    color=color,
                    label=label,
                )
                ax[2].scatter(
                    n_star, alpha_star, marker=marker, color=color, label=label
                )

        ax[0].set_xlabel(r"$\Delta^2_\star$", fontsize=fontsize)
        ax[0].set_ylabel(r"$n_\star$", fontsize=fontsize)
        ax[0].legend()
        if suite_emu != "mpg":
            ax[1].set_xlabel(r"$\Delta^2_\star$", fontsize=fontsize)
            ax[1].set_ylabel(r"$\alpha_\star$", fontsize=fontsize)
            ax[2].set_xlabel(r"$n_\star$", fontsize=fontsize)
            ax[2].set_ylabel(r"$\alpha_\star$", fontsize=fontsize)
        plt.tight_layout()

        if self.save_directory is not None:
            name = self.save_directory + "/cosmo_mle"
            plt.savefig(name + ".pdf")
            plt.savefig(name + ".png")
        else:
            plt.show()

    def plot_corner_chainconsumer(
        self,
        plot_params=None,
        delta_lnprob_cut=None,
        usetex=True,
        serif=True,
        only_cosmo=False,
        extra_nburn=0,
    ):
        """Make corner plot in ChainConsumer
        - plot_params: Pass a list of parameters to plot (in LaTeX form),
                    or leave as None to
                    plot all (including derived)
        - if delta_lnprob_cut is set, keep only high-prob points"""

        from chainconsumer import ChainConsumer, Chain, Truth
        import pandas as pd

        params_plot, strings_plot, _ = self.fitter.get_all_params(
            delta_lnprob_cut=delta_lnprob_cut, extra_nburn=extra_nburn
        )
        if only_cosmo:
            yesplot = ["$\\Delta^2_\\star$", "$n_\\star$"]
            for param in self.fitter.like.free_params:
                if param.name == "nrun":
                    yesplot.append("$\\alpha_\\star$")
        else:
            diff = np.max(params_plot, axis=0) - np.min(params_plot, axis=0)
            yesplot = np.array(strings_plot)[diff != 0]

        dict_pd = {}
        for ii, par in enumerate(strings_plot):
            if par in yesplot:
                dict_pd[par] = params_plot[:, ii]
        pd_data = pd.DataFrame(data=dict_pd)

        c = ChainConsumer()
        chain = Chain(
            samples=pd_data,
            name="a",
        )
        c.add_chain(chain)
        # summary = c.analysis.get_summary()["a"]
        if self.fitter.truth is not None:
            c.add_truth(
                Truth(
                    location=self.fitter.truth,
                    line_style="--",
                    color="C1",
                )
            )
        c.add_truth(
            Truth(
                location=self.fitter.mle,
                line_style=":",
                color="C2",
            )
        )

        fig = c.plotter.plot(figsize=(12, 12))

        if self.save_directory is not None:
            if only_cosmo:
                name = self.save_directory + "/corner_cosmo"
            else:
                name = self.save_directory + "/corner"
            plt.savefig(name + ".pdf")
            plt.savefig(name + ".png")

        # return summary

    def plot_corner(
        self,
        delta_lnprob_cut=None,
        usetex=True,
        only_cosmo=False,
        extra_nburn=0,
        only_cosmo_lims=True,
        extra_data=None,
    ):
        """Make corner plot in corner"""

        params_plot, strings_plot, _ = self.fitter.get_all_params(
            delta_lnprob_cut=delta_lnprob_cut, extra_nburn=extra_nburn
        )
        if only_cosmo:
            yesplot = ["$\\Delta^2_\\star$", "$n_\\star$"]
            if "nrun" in self.fitter.like.free_param_names:
                yesplot.append("$\\alpha_\\star$")
        else:
            # only plot parameters that vary
            diff = np.max(params_plot, axis=0) - np.min(params_plot, axis=0)
            yesplot = np.array(strings_plot)[diff != 0]

        truth = np.zeros((len(yesplot)))
        MLE = np.zeros((len(yesplot)))
        chain = np.zeros((params_plot.shape[0], len(yesplot)))
        for ii, par in enumerate(yesplot):
            _ = np.argwhere(np.array(strings_plot) == par)[0, 0]
            chain[:, ii] = params_plot[:, _]
            if self.fitter.truth is not None:
                if par in self.fitter.truth:
                    truth[ii] = self.fitter.truth[par]
            if par in self.fitter.mle:
                MLE[ii] = self.fitter.mle[par]
            if par == "$A_s$":
                chain[:, ii] *= 1e9
                truth[ii] *= 1e9
                MLE[ii] *= 1e9
                yesplot[ii] = r"$A_s\times10^9$"

        fig = corner(
            chain,
            labels=yesplot,
            quantiles=(0.16, 0.5, 0.84),
            levels=(0.68, 0.95),
            show_titles=True,
            title_quantiles=(0.16, 0.5, 0.84),
            title_fmt=".4f",
            plot_datapoints=False,
            plot_density=False,
        )

        # add truth and MLE
        value1 = truth.copy()
        value2 = MLE.copy()
        ndim = len(value1)
        axes = np.array(fig.axes).reshape((ndim, ndim))
        for i in range(ndim):
            ax = axes[i, i]
            if self.fitter.truth is not None:
                ax.axvline(value1[i], color="C0", label="Truth")
            ax.axvline(value2[i], color="C2", label="MAP", linestyle=":")

            # Set up x limits
            xlim = np.array(ax.get_xlim())
            if self.fitter.truth is not None:
                val_min = np.nanmin([value1[i], value2[i]])
                val_max = np.nanmax([value1[i], value2[i]])
            else:
                val_min = np.nanmin([value2[i]])
                val_max = np.nanmax([value2[i]])

            if (xlim[0] > val_min) and np.isfinite(val_min):
                xlim[0] = val_min
            if (xlim[1] < val_max) and np.isfinite(val_max):
                xlim[1] = val_max
            xdiff = xlim[1] - xlim[0]
            ax.set_xlim(xlim[0] - 0.05 * xdiff, xlim[1] + 0.05 * xdiff)
        axes[0, 0].legend(loc="upper left")

        # Loop over the histograms
        for yi in range(ndim):
            for xi in range(yi):
                ax = axes[yi, xi]

                if self.fitter.truth is not None:
                    ax.axvline(value1[xi], color="C0")
                    ax.axhline(value1[yi], color="C0")
                    ax.plot(value1[xi], value1[yi], ".C0")

                ax.axvline(value2[xi], color="C2", linestyle=":")
                ax.axhline(value2[yi], color="C2", linestyle=":")
                ax.plot(value2[xi], value2[yi], ".C2", linestyle=":")

                # Set up x and y limits
                xlim = np.array(ax.get_xlim())
                if self.fitter.truth is not None:
                    val_min = np.min([value1[xi], value2[xi]])
                    val_max = np.max([value1[xi], value2[xi]])
                else:
                    val_min = np.min([value2[xi]])
                    val_max = np.max([value2[xi]])
                if (xlim[0] > val_min) and np.isfinite(val_min):
                    xlim[0] = val_min
                if (xlim[1] < val_max) and np.isfinite(val_max):
                    xlim[1] = val_max
                xdiff = xlim[1] - xlim[0]
                ax.set_xlim(xlim[0] - 0.05 * xdiff, xlim[1] + 0.05 * xdiff)

                ylim = np.array(ax.get_ylim())
                if self.fitter.truth is not None:
                    val_min = np.min([value1[yi], value2[yi]])
                    val_max = np.max([value1[yi], value2[yi]])
                else:
                    val_min = np.min([value2[yi]])
                    val_max = np.max([value2[yi]])
                if (ylim[0] > val_min) and np.isfinite(val_min):
                    ylim[0] = val_min
                if (ylim[1] < val_max) and np.isfinite(val_max):
                    ylim[1] = val_max
                ydiff = ylim[1] - ylim[0]
                ax.set_ylim(ylim[0] - 0.05 * ydiff, ylim[1] + 0.05 * ydiff)

        if (extra_data is not None) and only_cosmo:
            axes[1, 0].scatter(
                extra_data[:, 0], extra_data[:, 1], color="C3", s=2
            )
            if "nrun" in self.fitter.like.free_param_names:
                axes[2, 0].scatter(
                    extra_data[:, 0], extra_data[:, 2], color="C3", s=2
                )
                axes[2, 1].scatter(
                    extra_data[:, 1], extra_data[:, 2], color="C3", s=2
                )

        if only_cosmo:
            (
                labs,
                delta2_star,
                n_star,
                alpha_star,
                suite_emu,
                data_cosmo,
            ) = self.get_hc_star()

            if "nrun" in self.fitter.like.free_param_names:
                axs = np.array(fig.axes).reshape((3, 3))
                nproj = 3
            else:
                axs = np.array(fig.axes).reshape((2, 2))
                nproj = 1

            for ii in range(nproj):
                if ii == 0:
                    x = delta2_star
                    y = n_star
                    _ = np.argwhere(np.array(yesplot) == "$n_\\star$")[0, 0]
                    ychain = chain[:, _]
                    ax = axs[1, 0]
                elif ii == 1:
                    x = delta2_star
                    y = alpha_star
                    _ = np.argwhere(np.array(yesplot) == "$\\alpha_\\star$")[
                        0, 0
                    ]
                    ychain = chain[:, _]
                    ax = axs[2, 0]
                else:
                    x = n_star
                    y = alpha_star
                    _ = np.argwhere(np.array(yesplot) == "$\\alpha_\\star$")[
                        0, 0
                    ]
                    ychain = chain[:, _]
                    ax = axs[2, 1]

                ax.scatter(x, y, marker="o", color="C1", alpha=0.5)

                if only_cosmo_lims:
                    diff = 0.05 * (y.max() - y.min())
                    ax.set_ylim(
                        np.min([y.min(), ychain.min()]) - diff,
                        np.max([y.max(), ychain.max()]) + diff,
                    )

            for ii in range(len(axs)):
                if ii == 0:
                    x = delta2_star
                    _ = np.argwhere(np.array(yesplot) == "$\\Delta^2_\\star$")[
                        0, 0
                    ]
                    xchain = chain[:, _]
                elif ii == 1:
                    x = n_star
                    _ = np.argwhere(np.array(yesplot) == "$n_\\star$")[0, 0]
                    xchain = chain[:, _]
                else:
                    x = alpha_star
                    _ = np.argwhere(np.array(yesplot) == "$\\alpha_\\star$")[
                        0, 0
                    ]
                    xchain = chain[:, _]

                if only_cosmo_lims:
                    for jj in range(len(axs)):
                        diff = 0.05 * (x.max() - x.min())
                        axs[jj, ii].set_xlim(
                            np.min([x.min(), xchain.min()]) - diff,
                            np.max([x.max(), xchain.max()]) + diff,
                        )

        if self.save_directory is not None:
            if only_cosmo:
                if only_cosmo_lims:
                    name = self.save_directory + "/corner_cosmo_full"
                else:
                    name = self.save_directory + "/corner_cosmo"
            else:
                name = self.save_directory + "/corner"
            plt.savefig(name + ".pdf")
            plt.savefig(name + ".png")

    def plot_lnprob(self, extra_nburn=0):
        """Plot lnprob"""

        mask, _ = purge_chains(self.fitter.lnprob[extra_nburn:, :])

        for ii in range(self.fitter.lnprob.shape[1]):
            if ii in mask:
                plt.plot(self.fitter.lnprob[extra_nburn:, ii], alpha=0.5)
            else:
                plt.plot(self.fitter.lnprob[extra_nburn:, ii], "--", alpha=0.01)

        if self.save_directory is not None:
            plt.savefig(self.save_directory + "/lnprob.pdf")

    def plot_p1d(
        self,
        values=None,
        plot_every_iz=1,
        residuals=False,
        rand_posterior=None,
        stat_best_fit="mle",
    ):
        """Plot the P1D of the data and the emulator prediction
        for the MCMC best fit
        """

        ## Get best fit values for each parameter
        if values is None:
            values = self.mle_values

        if self.save_directory is not None:
            if rand_posterior is None:
                fname = "P1D_mle"
            else:
                fname = "best_fit_" + stat_best_fit + "_err_posterior"
            if residuals:
                plot_fname = self.save_directory + "/" + fname + "_residuals"
            else:
                plot_fname = self.save_directory + "/" + fname
        else:
            plot_fname = None

        self.fitter.like.plot_p1d(
            values=values,
            plot_every_iz=plot_every_iz,
            residuals=residuals,
            rand_posterior=rand_posterior,
            plot_fname=plot_fname,
        )

    def plot_P1D_initial(self, plot_every_iz=1, residuals=False):
        """Plot the P1D of the data and the emulator prediction
        for the fiducial model"""

        if self.save_directory is not None:
            if residuals:
                plot_fname = self.save_directory + "/P1D_initial_residuals"
            else:
                plot_fname = self.save_directory + "/P1D_initial"
        else:
            plot_fname = None

        self.fitter.like.plot_p1d(
            values=None,
            plot_every_iz=plot_every_iz,
            residuals=residuals,
            plot_fname=plot_fname,
        )

    def plot_histograms(self, cube=False, delta_lnprob_cut=None):
        """Make histograms for all dimensions, using re-normalized values if
        cube=True
        - if delta_lnprob_cut is set, use only high-prob points"""

        # get chain (from sampler or from file)
        chain, lnprob, blobs = self.fitter.get_chain(
            delta_lnprob_cut=delta_lnprob_cut
        )
        plt.figure()

        for ip in range(self.fitter.ndim):
            param = self.fitter.like.free_params[ip]
            if cube:
                values = chain[:, ip]
                title = param.name + " in cube"
            else:
                cube_values = chain[:, ip]
                values = param.value_from_cube(cube_values)
                title = param.name

            plt.hist(values, 100, color="k", histtype="step")
            plt.title(title)
            plt.show()

        return

    def plot_igm(
        self,
        value=None,
        rand_sample=None,
        stat_best_fit="mle",
        cloud=True,
    ):
        """Plot IGM histories"""

        if value is None:
            value = self.mle_values

        self.fitter.like.plot_igm(
            cloud=cloud,
            free_params=self.like_params,
            stat_best_fit=stat_best_fit,
            save_directory=self.save_directory,
        )

        # if rand_sample is not None:
        #     # chain, lnprob, blobs = self.fitter.get_chain()
        #     # nn = min(chain.shape[0], nn)
        #     # mask = np.random.permutation(chain.shape[0])[:nn]
        #     # rand_sample = chain[mask]

        #     # sample the chain to get errors on IGM parameters
        #     pars_samp = {}
        #     pars_samp["tau_eff"] = np.zeros((nn, len(z)))
        #     pars_samp["gamma"] = np.zeros((nn, len(z)))
        #     pars_samp["sigT_kms"] = np.zeros((nn, len(z)))
        #     pars_samp["kF_kms"] = np.zeros((nn, len(z)))
        #     for ii in range(nn):
        #         like_params = self.fitter.like.parameters_from_sampling_point(
        #             rand_sample[ii]
        #         )
        #         models = self.fitter.like.theory.update_igm_models(like_params)
        #         pars_samp["tau_eff"][ii] = models["F_model"].get_tau_eff(z)
        #         pars_samp["gamma"][ii] = models["T_model"].get_gamma(z)
        #         pars_samp["sigT_kms"][ii] = models["T_model"].get_sigT_kms(z)
        #         pars_samp["kF_kms"][ii] = models["P_model"].get_kF_kms(z)

        #     if rand_sample is not None:
        #         err = np.abs(
        #             np.percentile(pars_samp[arr_labs[ii]], [16, 84], axis=0)
        #             - pars_best[arr_labs[ii]]
        #         )
        #         ax[ii].errorbar(
        #             pars_best["z"],
        #             pars_best[arr_labs[ii]],
        #             err,
        #             label="best-fitting",
        #             alpha=0.75,
        #         )

    def compare_corners(
        self,
        chain_files,
        labels,
        plot_params=None,
        save_string=None,
        rootdir=None,
        subfolder=None,
        delta_lnprob_cut=None,
        usetex=True,
        serif=True,
    ):
        """Function to take a list of chain files and overplot the chains
        Pass a list of chain files (ints) and a list of labels (strings)
         - plot_params: list of parameters (in code variables, not latex form)
                        to plot if only a subset is desired
         - save_string: to save the plot. Must include
                        file extension (i.e. .pdf, .png etc)
         - if delta_lnprob_cut is set, keep only high-prob points"""

        from chainconsumer import ChainConsumer, Chain, Truth

        assert len(chain_files) == len(labels)

        truth_dict = {}
        c = ChainConsumer()

        ## Add each chain we want to plot
        for aa, chain_file in enumerate(chain_files):
            sampler = EmceeSampler(
                read_chain_file=chain_file, subfolder=subfolder, rootdir=rootdir
            )
            params, strings, _ = sampler.get_all_params(
                delta_lnprob_cut=delta_lnprob_cut
            )
            c.add_chain(params, parameters=strings, name=labels[aa])

            ## Do not check whether truth results are the same for now
            ## Take the longest truth dictionary for disjoint chains
            if len(sampler.truth) > len(truth_dict):
                truth_dict = sampler.truth

        c.configure(
            diagonal_tick_labels=False,
            tick_font_size=15,
            label_font_size=25,
            max_ticks=4,
            usetex=usetex,
            serif=serif,
        )

        if plot_params == None:
            fig = c.plotter.plot(figsize=(15, 15), truth=truth_dict)
        else:
            ## From plot_param list, build list of parameter
            ## strings to plot
            plot_param_strings = []
            for par in plot_params:
                plot_param_strings.append(self.fitter.param_dict[par])
            fig = c.plotter.plot(
                figsize=(10, 10),
                parameters=plot_param_strings,
                truth=truth_dict,
            )
        if save_string:
            fig.savefig("%s" % save_string)
        fig.show()

        return

    def plot_hcd_cont(
        self,
        plot_every_iz=1,
        smooth_k=False,
        plot_data=False,
        zrange=[0, 10],
    ):
        """Function to plot the HCD contamination"""

        if plot_data:
            dict_data = self.mle_results
        else:
            dict_data = None

        list_params = {}
        Npar_damp = 0
        Npar_scale = 0
        for p in self.fitter.like.free_params:
            if ("A_damp" in p.name) | ("A_scale" in p.name):
                key = self.fitter.param_dict[p.name]
                list_params[p.name] = self.fitter.mle[key]
                print(p.name, self.fitter.mle[key])
            if "A_damp" in p.name:
                Npar_damp += 1
            elif "A_scale" in p.name:
                Npar_scale += 1

        Npar = len(list_params)
        if Npar == 0:
            return

        print(Npar, Npar_damp, Npar_scale)

        if Npar_scale != 0:
            A_scale_coeff = np.zeros(Npar_scale)
            for ii in range(Npar_scale):
                name = "ln_A_scale_" + str(ii)
                # note non-trivial order in coefficients
                A_scale_coeff[Npar_scale - ii - 1] = list_params[name]
        else:
            A_scale_coeff = None

        if Npar_damp != 0:
            A_damp_coeff = np.zeros(Npar_damp)
            for ii in range(Npar_damp):
                name = "ln_A_damp_" + str(ii)
                # note non-trivial order in coefficients
                A_damp_coeff[Npar_damp - ii - 1] = list_params[name]
        else:
            A_damp_coeff = None

        if self.save_directory is not None:
            name = self.save_directory + "/HCD_cont"
        else:
            name = None

        self.fitter.like.theory.model_cont.hcd_model.plot_contamination(
            self.fitter.like.data.z,
            self.fitter.like.data.k_kms,
            ln_A_damp_coeff=A_damp_coeff,
            ln_A_scale_coeff=A_scale_coeff,
            plot_every_iz=plot_every_iz,
            cmap=self.cmap,
            smooth_k=smooth_k,
            dict_data=dict_data,
            zrange=zrange,
            name=name,
        )

    def plot_metal_cont(
        self,
        plot_every_iz=1,
        stat_best_fit="mle",
        smooth_k=False,
        plot_data=False,
        zrange=[0, 10],
    ):
        """Function to plot metal contamination"""

        if plot_data:
            dict_data = self.mle_results
        else:
            dict_data = None

        # get mean flux from best-fitting model
        mF = self.fitter.like.theory.model_igm.F_model.get_mean_flux(
            np.array(self.fitter.like.data.z), self.like_params
        )

        # plot contamination of all metals
        metal_models = self.fitter.like.theory.model_cont.metal_models
        for jj in range(len(metal_models)):
            metal = metal_models[jj].metal_label

            x_list_params = {}
            d_list_params = {}
            a_list_params = {}
            for p in self.fitter.like.free_params:
                if "ln_x_" + metal + "_" in p.name:
                    key = self.fitter.param_dict[p.name]
                    x_list_params[p.name] = self.fitter.mle[key]
                    print(p.name, self.fitter.mle[key])
                if "ln_d_" + metal + "_" in p.name:
                    key = self.fitter.param_dict[p.name]
                    d_list_params[p.name] = self.fitter.mle[key]
                    print(p.name, self.fitter.mle[key])
                if "a_" + metal + "_" in p.name:
                    key = self.fitter.param_dict[p.name]
                    a_list_params[p.name] = self.fitter.mle[key]
                    print(p.name, self.fitter.mle[key])

            x_Npar = len(x_list_params)
            ln_X_coeff = np.zeros(x_Npar)
            for ii in range(x_Npar):
                name = "ln_x_" + metal + "_" + str(ii)
                # note non-trivial order in coefficients
                ln_X_coeff[x_Npar - ii - 1] = x_list_params[name]

            d_Npar = len(d_list_params)
            ln_D_coeff = np.zeros(d_Npar)
            for ii in range(d_Npar):
                name = "ln_d_" + metal + "_" + str(ii)
                # note non-trivial order in coefficients
                ln_D_coeff[d_Npar - ii - 1] = d_list_params[name]

            a_Npar = len(a_list_params)
            A_coeff = np.zeros(a_Npar)
            for ii in range(a_Npar):
                name = "a_" + metal + "_" + str(ii)
                # note non-trivial order in coefficients
                A_coeff[d_Npar - ii - 1] = a_list_params[name]

            if (x_Npar == 0) and (d_Npar == 0) and (a_Npar == 0):
                continue

            if x_Npar == 0:
                ln_X_coeff = None
            if d_Npar == 0:
                ln_D_coeff = None
            if a_Npar == 0:
                A_coeff = None

            if self.save_directory is not None:
                name = self.save_directory + "/" + metal + "_cont"
            else:
                name = None

            metal_models[jj].plot_contamination(
                self.fitter.like.data.z,
                self.fitter.like.data.k_kms,
                mF,
                ln_X_coeff=ln_X_coeff,
                ln_D_coeff=ln_D_coeff,
                A_coeff=A_coeff,
                plot_every_iz=plot_every_iz,
                cmap=self.cmap,
                smooth_k=smooth_k,
                dict_data=dict_data,
                zrange=zrange,
                name=name,
            )

    def plot_agn_cont(
        self,
        plot_every_iz=1,
        smooth_k=False,
        plot_data=False,
        zrange=[0, 10],
    ):
        """Function to plot AGN contamination"""

        if plot_data:
            dict_data = self.mle_results
        else:
            dict_data = None

        list_params = {}
        for p in self.fitter.like.free_params:
            if "ln_AGN" in p.name:
                key = self.fitter.param_dict[p.name]
                list_params[p.name] = self.fitter.mle[key]
                print(p.name, self.fitter.mle[key])

        Npar = len(list_params)
        if Npar == 0:
            return
        coeff = np.zeros(Npar)
        for ii in range(Npar):
            name = "ln_AGN_" + str(ii)
            # note non-trivial order in coefficients
            coeff[Npar - ii - 1] = list_params[name]

        if self.save_directory is not None:
            name = self.save_directory + "/AGN_cont"
        else:
            name = None

        self.fitter.like.theory.model_cont.agn_model.plot_contamination(
            self.fitter.like.data.z,
            self.fitter.like.data.k_kms,
            coeff,
            plot_every_iz=plot_every_iz,
            cmap=self.cmap,
            smooth_k=smooth_k,
            dict_data=dict_data,
            zrange=zrange,
            name=name,
        )

    def plot_hull(self, p0=None, save_plot=True):
        """Function to plot data within hull"""

        if p0 is None:
            p0 = self.fitter.mle_cube

        like_params = self.fitter.like.parameters_from_sampling_point(p0)

        emu_call, M_of_z, blob = self.fitter.like.theory.get_emulator_calls(
            self.fitter.like.data.z,
            like_params=like_params,
            return_M_of_z=True,
            return_blob=True,
        )

        p1 = np.zeros(
            (
                self.fitter.like.theory.hull.nz,
                len(self.fitter.like.theory.hull.params),
            )
        )
        for jj, key in enumerate(self.fitter.like.theory.hull.params):
            p1[:, jj] = emu_call[key]

        # print(self.fitter.like.theory.hull.in_hulls(p1))
        self.fitter.like.theory.hull.plot_hulls(p1)

        if self.save_directory is not None:
            name = self.save_directory + "/fit_in_hull"
            if save_plot:
                plt.savefig(name + ".pdf")
                plt.savefig(name + ".png")
        else:
            plt.show()
