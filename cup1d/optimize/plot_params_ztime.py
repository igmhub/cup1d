import numpy as np
import matplotlib.pyplot as plt


def plot_z_at_time_params(fitter, out_mle, save_fig=None):
    """Make the plot and get weak priors"""
    # paramstrings = [
    #     r"$\tau_{\rm eff_0}$",
    #     r"$\\sigma_{\rm T_0}$",
    #     r"$\\mathrm{ln}\\,f($\\mathrm{Ly}\x07lpha-\\mathrm{SiIII}$_0)$",
    #     r"$\\mathrm{ln}\\,s($\\mathrm{Ly}\x07lpha-\\mathrm{SiIII}$_0)$",
    #     r"$\\mathrm{ln}\\,f($\\mathrm{Ly}\x07lpha-\\mathrm{SiII}$_0)$",
    #     r"$\\mathrm{ln}\\,s($\\mathrm{Ly}\x07lpha-\\mathrm{SiII}$_0)$",
    #     r"$\\mathrm{ln}\\,f($\\mathrm{SiIIa}_\\mathrm{SiIIb}$_0)$",
    #     r"$\\mathrm{ln}\\,s($\\mathrm{SiIIa}_\\mathrm{SiIIb}$_0)$",
    #     r"$\\mathrm{ln}\\,f($\\mathrm{SiIIa}_\\mathrm{SiIII}$_0)$",
    #     r"$\\mathrm{ln}\\,f($\\mathrm{SiIIb}_\\mathrm{SiIII}$_0)$",
    #     r"$f_{\rm HCD1}_0$",
    #     r"$f_{\rm HCD4}_0$",
    # ]

    paramstrings = list(out_mle[0].keys())
    for key in ["Delta2_star", "n_star", "alpha_star"]:
        paramstrings.remove(key)

    ofit = {}
    # print(fitter.param_dict_rev)
    for par in paramstrings:
        ofit[fitter.param_dict_rev[par]] = 1

    weak_priors = {}

    fig, ax = plt.subplots(3, 4, figsize=(16, 16), sharex=True)
    ax = ax.reshape(-1)

    dict_out = {}
    jj = 0
    for ii, key in enumerate(paramstrings):
        if key not in out_mle[0]:
            continue
        dict_out[key] = np.zeros(len(out_mle))
        for iz in range(len(out_mle)):
            if key in out_mle[iz]:
                ax[jj].scatter(fitter.like.data.z[iz], out_mle[iz][key])
                dict_out[key][iz] = out_mle[iz][key]
        ax[jj].set_ylabel(fitter.param_dict_rev[key])
        ax[jj].set_xlabel(r"$z$")
        jj += 1

    jj = 0
    for ii, key in enumerate(paramstrings):
        if key not in dict_out:
            continue
        print(
            fitter.param_dict_rev[key],
            np.round(np.median(dict_out[key]), 4),
            np.round(np.std(dict_out[key]), 4),
        )
        ind = np.argwhere(dict_out[key] != 0)[:, 0]
        ax[jj].plot(
            fitter.like.data.z[ind],
            fitter.like.data.z[ind] * 0 + np.median(dict_out[key][ind]),
        )

        # ind = np.argwhere(fitter.param_dict_rev[key] == list_props)[0,0]
        # w = np.abs(delta_chi2[ind])
        x = fitter.like.data.z.copy()[ind]
        y = dict_out[key].copy()[ind]
        w = np.ones_like(x)
        fit = np.polyfit(x, y, ofit[fitter.param_dict_rev[key]], w=w)
        for kk in range(3):
            mod = np.poly1d(fit)(x)
            std_mod = np.std(mod - y)
            # if "ln_x_" in fitter.param_dict_rev[key]:
            #     _ = (np.abs(y - mod) < 2 * std_mod) & (y > -8)
            # else:
            _ = np.abs(y - mod) < 2 * std_mod
            x = x[_]
            y = y[_]
            w = w[_]
            fit = np.polyfit(x, y, ofit[fitter.param_dict_rev[key]], w=w)
        # ax[jj].plot(like.data.z, mod)
        mod = np.poly1d(fit)(fitter.like.data.z[ind])
        ax[jj].errorbar(fitter.like.data.z[ind], mod, std_mod * 2)
        print(
            np.round(np.min(mod - std_mod), 2),
            np.round(np.max(mod + std_mod), 2),
            fit,
        )
        weak_priors[fitter.param_dict_rev[key] + "_cen"] = mod
        weak_priors[fitter.param_dict_rev[key] + "_std"] = std_mod
        jj += 1

    plt.tight_layout()
    plt.show()
    if save_fig is not None:
        plt.savefig("snr3_fid_weakp.png")
        plt.savefig("snr3_fid_weakp.pdf")
    return weak_priors
