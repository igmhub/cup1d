import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import os
import lace


# Function to generate n discrete colors from any continuous colormap
def get_discrete_cmap(n, base_cmap="jet"):
    """Returns a colormap with n discrete colors."""
    cmap = plt.cm.get_cmap(
        base_cmap, n
    )  # Sample n colors from the base colormap
    return ListedColormap(cmap(np.linspace(0, 1, n)))


class Plotter(object):
    def __init__(self, fitter, save_directory=None):
        self.fitter = fitter
        self.cmap = get_discrete_cmap(len(self.fitter.like.data.z))
        self.save_directory = save_directory

        self.mle_values = self.fitter.get_best_fit(stat_best_fit="mle")
        self.like_params = self.fitter.like.parameters_from_sampling_point(
            self.mle_values
        )
        self.mle_results = self.fitter.like.plot_p1d(
            values=self.mle_values, plot_every_iz=1, return_all=True
        )
        plt.close()

    def plots_minimizer(self, zrange=[0, 10]):
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

    def plots_sampler(self):
        # plot lnprob
        self.plot_lnprob()

        # plot initial P1D (before fitting)
        self.plot_P1D_initial(residuals=False)
        self.plot_P1D_initial(residuals=True)

        # plot best fit
        self.plot_p1d(residuals=False, stat_best_fit="mle")
        plt.close()
        self.plot_p1d(residuals=True, stat_best_fit="mle")
        plt.close()

        # plot cosmology
        if self.fitter.fix_cosmology == False:
            self.plot_corner(only_cosmo=True)
        plt.close()

        # plot corner
        self.plot_corner()

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

    def plot_mle_cosmo(self, fontsize=16, nyx_version="Jul2024"):
        """Plot MLE cosmology"""

        suite_emu = self.fitter.like.theory.emulator.list_sim_cube[0][:3]
        if suite_emu == "mpg":
            repo = os.path.dirname(lace.__path__[0]) + "/"
            fname = repo + ("data/sim_suites/Australia20/mpg_emu_cosmo.npy")
        elif suite_emu == "nyx":
            fname = (
                os.environ["NYX_PATH"] + "nyx_emu_cosmo_" + nyx_version + ".npy"
            )
        else:
            ValueError("cosmo_label should be 'mpg' or 'nyx'")

        try:
            data_cosmo = np.load(fname, allow_pickle=True)
        except:
            ValueError(f"{fname} not found")

        labs = []
        delta2_star = np.zeros(len(data_cosmo))
        n_star = np.zeros(len(data_cosmo))
        alpha_star = np.zeros(len(data_cosmo))

        for ii in range(len(data_cosmo)):
            labs.append(data_cosmo[ii]["sim_label"])
            delta2_star[ii] = data_cosmo[ii]["star_params"]["Delta2_star"]
            n_star[ii] = data_cosmo[ii]["star_params"]["n_star"]
            alpha_star[ii] = data_cosmo[ii]["star_params"]["alpha_star"]

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
            if data_cosmo[ii]["sim_label"][-1].isdigit():
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

        ax[0].scatter(
            self.fitter.mle_cosmo["Delta2_star"],
            self.fitter.mle_cosmo["n_star"],
            marker="X",
            color="C1",
        )
        if suite_emu != "mpg":
            ax[1].scatter(
                self.fitter.mle_cosmo["Delta2_star"],
                self.fitter.mle_cosmo["alpha_star"],
                marker="X",
                color="C1",
            )
            ax[2].scatter(
                self.fitter.mle_cosmo["n_star"],
                self.fitter.mle_cosmo["alpha_star"],
                marker="X",
                color="C1",
            )

        ax[0].set_xlabel(r"$\Delta^2_\star$", fontsize=fontsize)
        ax[0].set_ylabel(r"$n_\star$", fontsize=fontsize)
        if suite_emu != "mpg":
            ax[1].set_xlabel(r"$\Delta^2_\star$", fontsize=fontsize)
            ax[1].set_ylabel(r"$\alpha_\star$", fontsize=fontsize)
            ax[2].set_xlabel(r"$n_\star$", fontsize=fontsize)
            ax[2].set_ylabel(r"$\alpha_\star$", fontsize=fontsize)
        plt.tight_layout()

        if self.save_directory is not None:
            plt.savefig(self.save_directory + "/cosmo_mle.pdf")
        else:
            plt.show()

    def plot_corner(
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
        summary = c.analysis.get_summary()["a"]
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
                plt.savefig(self.save_directory + "/corner_cosmo.pdf")
            else:
                plt.savefig(self.save_directory + "/corner.pdf")

        return summary

    def plot_lnprob(self, extra_nburn=0):
        """Plot lnprob"""

        mask, _ = purge_chains(self.fitter.lnprob[extra_nburn:, :])
        mask_use = (
            "Using "
            + str(mask.shape[0])
            + " chains out of "
            + str(self.fitter.lnprob.shape[1])
        )

        for ii in range(self.fitter.lnprob.shape[1]):
            if ii in mask:
                plt.plot(self.fitter.lnprob[extra_nburn:, ii], alpha=0.5)
            # else:
            #     plt.plot(self.fitter.lnprob[extra_nburn:, ii], "--", alpha=0.5)

        if self.save_directory is not None:
            plt.savefig(self.save_directory + "/lnprob.pdf")
        # plt.close()

        return mask_use

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
                plot_fname = (
                    self.save_directory + "/" + fname + "_residuals.pdf"
                )
            else:
                plot_fname = self.save_directory + "/" + fname + ".pdf"
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
                plot_fname = self.save_directory + "/P1D_initial_residuals.pdf"
            else:
                plot_fname = self.save_directory + "/P1D_initial.pdf"
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
        cloud=False,
    ):
        """Plot IGM histories"""

        if value is None:
            value = self.mle_values

        # true IGM parameters
        if self.fitter.like.truth is not None:
            pars_true = {}
            pars_true["z"] = self.fitter.like.truth["igm"]["z"]
            pars_true["tau_eff"] = self.fitter.like.truth["igm"]["tau_eff"]
            pars_true["gamma"] = self.fitter.like.truth["igm"]["gamma"]
            pars_true["sigT_kms"] = self.fitter.like.truth["igm"]["sigT_kms"]
            pars_true["kF_kms"] = self.fitter.like.truth["igm"]["kF_kms"]

        # fiducial IGM parameters
        pars_fid = {}
        pars_fid["z"] = self.fitter.like.fid["igm"]["z"]
        pars_fid["tau_eff"] = self.fitter.like.fid["igm"]["tau_eff"]
        pars_fid["gamma"] = self.fitter.like.fid["igm"]["gamma"]
        pars_fid["sigT_kms"] = self.fitter.like.fid["igm"]["sigT_kms"]
        pars_fid["kF_kms"] = self.fitter.like.fid["igm"]["kF_kms"]

        # all IGM histories in the training sample
        if cloud:
            all_emu_igm = self.fitter.like.theory.model_igm.all_igm

        pars_best = {}
        pars_best["z"] = np.array(self.fitter.like.data.z)
        pars_best[
            "tau_eff"
        ] = self.fitter.like.theory.model_igm.F_model.get_tau_eff(
            pars_best["z"], like_params=self.like_params
        )
        pars_best[
            "gamma"
        ] = self.fitter.like.theory.model_igm.T_model.get_gamma(
            pars_best["z"], like_params=self.like_params
        )
        pars_best[
            "sigT_kms"
        ] = self.fitter.like.theory.model_igm.T_model.get_sigT_kms(
            pars_best["z"], like_params=self.like_params
        )
        pars_best[
            "kF_kms"
        ] = self.fitter.like.theory.model_igm.P_model.get_kF_kms(
            pars_best["z"], like_params=self.like_params
        )

        if rand_sample is not None:
            # chain, lnprob, blobs = self.fitter.get_chain()
            # nn = min(chain.shape[0], nn)
            # mask = np.random.permutation(chain.shape[0])[:nn]
            # rand_sample = chain[mask]

            # sample the chain to get errors on IGM parameters
            pars_samp = {}
            pars_samp["tau_eff"] = np.zeros((nn, len(z)))
            pars_samp["gamma"] = np.zeros((nn, len(z)))
            pars_samp["sigT_kms"] = np.zeros((nn, len(z)))
            pars_samp["kF_kms"] = np.zeros((nn, len(z)))
            for ii in range(nn):
                like_params = self.fitter.like.parameters_from_sampling_point(
                    rand_sample[ii]
                )
                models = self.fitter.like.theory.update_igm_models(like_params)
                pars_samp["tau_eff"][ii] = models["F_model"].get_tau_eff(z)
                pars_samp["gamma"][ii] = models["T_model"].get_gamma(z)
                pars_samp["sigT_kms"][ii] = models["T_model"].get_sigT_kms(z)
                pars_samp["kF_kms"][ii] = models["P_model"].get_kF_kms(z)

        # plot the IGM histories
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
            if self.fitter.like.truth is not None:
                _ = pars_true[arr_labs[ii]] != 0
                ax[ii].plot(
                    pars_true["z"][_],
                    pars_true[arr_labs[ii]][_],
                    ":o",
                    label="true",
                    alpha=0.75,
                )
            _ = pars_fid[arr_labs[ii]] != 0
            ax[ii].plot(
                pars_fid["z"][_],
                pars_fid[arr_labs[ii]][_],
                "--",
                label="fiducial",
                lw=3,
                alpha=0.75,
            )

            _ = pars_best[arr_labs[ii]] != 0
            if rand_sample is not None:
                err = np.abs(
                    np.percentile(pars_samp[arr_labs[ii]], [16, 84], axis=0)
                    - pars_best[arr_labs[ii]]
                )
                ax[ii].errorbar(
                    pars_best["z"],
                    pars_best[arr_labs[ii]],
                    err,
                    label="best-fitting",
                    alpha=0.75,
                )
            else:
                ax[ii].plot(
                    pars_best["z"][_],
                    pars_best[arr_labs[ii]][_],
                    label="fit",
                    lw=3,
                    alpha=0.75,
                )

            if cloud:
                for sim_label in all_emu_igm:
                    _ = all_emu_igm[sim_label][arr_labs[ii]] != 0
                    ax[ii].scatter(
                        all_emu_igm[sim_label]["z"][_],
                        all_emu_igm[sim_label][arr_labs[ii]][_],
                        marker=".",
                        color="black",
                        alpha=0.05,
                    )

            ax[ii].set_ylabel(latex_labs[ii])
            if ii == 0:
                ax[ii].set_yscale("log")
                ax[ii].legend()

            if (ii == 2) | (ii == 3):
                ax[ii].set_xlabel(r"$z$")

        plt.tight_layout()

        if self.save_directory is not None:
            plt.savefig(
                self.save_directory + "/IGM_histories_" + stat_best_fit + ".pdf"
            )
        else:
            plt.show()

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
        for p in self.fitter.like.free_params:
            if "A_damp" in p.name:
                key = self.fitter.param_dict[p.name]
                list_params[p.name] = self.fitter.mle[key]
                print(p.name, self.fitter.mle[key])

        Npar = len(list_params)
        if Npar == 0:
            return
        coeff = np.zeros(Npar)
        for ii in range(Npar):
            name = "ln_A_damp_" + str(ii)
            # note non-trivial order in coefficients
            coeff[Npar - ii - 1] = list_params[name]

        self.fitter.like.theory.model_cont.hcd_model.plot_contamination(
            self.fitter.like.data.z,
            self.fitter.like.data.k_kms,
            coeff,
            plot_every_iz=plot_every_iz,
            cmap=self.cmap,
            smooth_k=smooth_k,
            dict_data=dict_data,
            zrange=zrange,
        )

        if self.save_directory is not None:
            plt.savefig(self.save_directory + "/HCD_cont.pdf")
        else:
            plt.show()

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
            for p in self.fitter.like.free_params:
                if "ln_" + metal + "_" in p.name:
                    key = self.fitter.param_dict[p.name]
                    x_list_params[p.name] = self.fitter.mle[key]
                    print(p.name, self.fitter.mle[key])
                if "d_" + metal + "_" in p.name:
                    key = self.fitter.param_dict[p.name]
                    d_list_params[p.name] = self.fitter.mle[key]
                    print(p.name, self.fitter.mle[key])

            x_Npar = len(x_list_params)
            ln_X_coeff = np.zeros(x_Npar)
            for ii in range(x_Npar):
                name = "ln_" + metal + "_" + str(ii)
                # note non-trivial order in coefficients
                ln_X_coeff[x_Npar - ii - 1] = x_list_params[name]

            d_Npar = len(d_list_params)
            ln_D_coeff = np.zeros(d_Npar)
            for ii in range(d_Npar):
                name = "d_" + metal + "_" + str(ii)
                # note non-trivial order in coefficients
                ln_D_coeff[d_Npar - ii - 1] = d_list_params[name]

            if (x_Npar == 0) and (d_Npar == 0):
                continue

            if x_Npar == 0:
                ln_X_coeff = None
            if d_Npar == 0:
                ln_D_coeff = None

            print(ln_X_coeff, ln_D_coeff)

            metal_models[jj].plot_contamination(
                self.fitter.like.data.z,
                self.fitter.like.data.k_kms,
                mF,
                ln_X_coeff=ln_X_coeff,
                ln_D_coeff=ln_D_coeff,
                plot_every_iz=plot_every_iz,
                cmap=self.cmap,
                smooth_k=smooth_k,
                dict_data=dict_data,
                zrange=zrange,
            )

            if self.save_directory is not None:
                plt.savefig(self.save_directory + "/" + metal + "_cont.pdf")
                plt.close()
            else:
                plt.show()

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

        self.fitter.like.theory.model_cont.agn_model.plot_contamination(
            self.fitter.like.data.z,
            self.fitter.like.data.k_kms,
            coeff,
            plot_every_iz=plot_every_iz,
            cmap=self.cmap,
            smooth_k=smooth_k,
            dict_data=dict_data,
            zrange=zrange,
        )

        if self.save_directory is not None:
            plt.savefig(self.save_directory + "/AGN_cont.pdf")
        else:
            plt.show()