"""Plots for importance sampling external chains with DESI P1D constraints."""

from getdist import plots
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

from cup1d.likelihood.marginal import gaussian_chi2


_PARAMETER_LABELS = {
    "mnu": r"$\sum m_\nu$",
    "nnu": r"$N_\mathrm{eff}$",
    "nrun": r"$\alpha_\mathrm{s}$",
    "nrunrun": r"$\beta_\mathrm{s}$",
    "tau": r"$\tau$",
    "omegak": r"$\Omega_k$",
    "r": r"$r$",
    "w": r"$w_0$",
    "wa": r"$w_a$",
}


def plot_importance_sampling(
    chains,
    desi_constraint,
    parameter_name=None,
    *,
    fontsize=24,
    include_original_second_chain=False,
    use_as_ns=False,
    additional_parameters=(),
    desi_label=r"DESI $P_\mathrm{1D}$",
    save_path=None,
):
    """Plot CMB chains before and after DESI P1D importance sampling.

    Parameters
    ----------
    chains
        Ordered dictionaries returned by the chain loaders. The first is the
        Planck chain; an optional second is the CMB-SPA chain. The samples are
        copied before plotting or reweighting, so this function never changes
        the input chains.
    desi_constraint
        Dictionary containing ``Delta2_star``, ``n_star``, their uncertainties,
        and correlation ``r``.
    parameter_name
        Optional cosmological parameter to show. When omitted, the plot shows
        only the two linear-power parameters, or ``logA`` and ``ns`` when
        ``use_as_ns=True``.
    additional_parameters
        Further cosmological parameters to append to the triangle plot. This
        is useful for extensions with more than one parameter, such as
        ``("w", "wa")`` for $w_0w_a$CDM.
    desi_label
        Label used for the reweighted DESI constraint in the legend.
    include_original_second_chain
        Include the unweighted second chain in addition to its reweighted
        version. ``use_as_ns`` selects ``logA`` and ``ns`` rather than the
        linear-power amplitude and slope.
    save_path
        Optional path without an extension. Both PNG and PDF are written.

    Returns
    -------
    getdist.plots.GetDistPlotter
        The configured plotter, including the reweighted chains.
    """
    if parameter_name is not None and parameter_name not in _PARAMETER_LABELS:
        raise ValueError(f"Unsupported parameter: {parameter_name}")
    unknown_parameters = set(additional_parameters) - _PARAMETER_LABELS.keys()
    if unknown_parameters:
        raise ValueError(f"Unsupported parameters: {sorted(unknown_parameters)}")

    plotted_chains = []
    labels = []
    colors = ["C4", "C2", "C1", "C0"]
    line_styles = [":", "-.", "--", "-"]

    for index, chain in enumerate(chains):
        original = chain["samples"].copy()
        _ensure_ranges_periodic(original)
        reweighted = original.copy()
        _ensure_ranges_periodic(reweighted)
        params = reweighted.getParams()
        log_likelihood = 0.5 * gaussian_chi2(
            params.linP_n_star,
            params.linP_DL2_star,
            desi_constraint["n_star"],
            desi_constraint["Delta2_star"],
            desi_constraint["n_star_err"],
            desi_constraint["Delta2_star_err"],
            desi_constraint["r"],
        )
        reweighted.reweightAddingLogLikes(log_likelihood)
        if index == 0:
            plotted_chains.append(original)
            labels.append(r"$\mathit{Planck}$ T&E")
        elif index == 1 and include_original_second_chain:
            plotted_chains.append(original)
            labels.append(r"CMB-SPA + DESI BAO")

        plotted_chains.append(reweighted)
        if index == 0:
            labels.append(r"$\mathit{Planck}$ T&E + " + desi_label)
        else:
            labels.append(
                r"CMB-SPA + DESI BAO" + "\n+ " + desi_label
            )

    _print_chain_summaries(
        plotted_chains, parameter_name, use_as_ns, additional_parameters
    )

    if parameter_name == "nrunrun":
        parameters = ["logA", "ns", "nrun", parameter_name] if use_as_ns else [
            "linP_DL2_star", "linP_n_star", "nrun", parameter_name
        ]
    elif parameter_name is not None:
        parameters = ["logA", "ns", parameter_name] if use_as_ns else [
            "linP_DL2_star", "linP_n_star", parameter_name
        ]
    else:
        parameters = ["logA", "ns"] if use_as_ns else [
            "linP_DL2_star", "linP_n_star"
        ]
    parameters.extend(additional_parameters)

    plotter = plots.getSubplotPlotter(width_inch=8)
    plotter.settings.axes_fontsize = fontsize
    plotter.settings.legend_fontsize = fontsize
    count = len(labels)
    plotter.triangle_plot(
        plotted_chains,
        parameters,
        legend_labels=labels,
        legend_loc="upper right",
        colors=colors[:count],
        filled=True,
        lws=[3] * count,
        alphas=[0.8] * count,
        line_args=[
            {"ls": line_styles[index], "color": colors[index], "lw": 3, "alpha": 0.8}
            for index in range(count)
        ],
    )
    _format_plot(
        plotter, parameter_name, use_as_ns, additional_parameters, fontsize
    )
    if save_path is not None:
        plotter.fig.savefig(f"{save_path}.png", bbox_inches="tight")
        plotter.fig.savefig(f"{save_path}.pdf", bbox_inches="tight")
    return plotter


def _format_plot(
    plotter, parameter_name, use_as_ns, additional_parameters, fontsize
):
    """Apply the conventional axes, reference lines, and limits."""
    dimension = 2 + len(additional_parameters)
    if parameter_name is not None:
        dimension += 2 if parameter_name == "nrunrun" else 1
    for column in range(dimension):
        axis = plotter.subplots[-1, column]
        axis.tick_params(axis="both", which="major", labelsize=fontsize)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=3))
        axis.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    for row in range(1, dimension):
        axis = plotter.subplots[row, 0]
        axis.tick_params(axis="both", which="major", labelsize=fontsize)
        axis.yaxis.set_major_locator(MaxNLocator(nbins=3))
        axis.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    if use_as_ns:
        plotter.subplots[-1, 0].set_xlabel(r"$\log(10^{10} A_\mathrm{s})$", fontsize=fontsize)
        plotter.subplots[-1, 1].set_xlabel(r"$n_\mathrm{s}$", fontsize=fontsize)
        plotter.subplots[1, 0].set_ylabel(r"$n_\mathrm{s}$", fontsize=fontsize)
    else:
        plotter.subplots[-1, 0].set_xlabel(r"$\Delta^2_\star$", fontsize=fontsize)
        plotter.subplots[-1, 1].set_xlabel(r"$n_\star$", fontsize=fontsize)
        plotter.subplots[1, 0].set_ylabel(r"$n_\star$", fontsize=fontsize)

    if parameter_name == "nrunrun":
        plotter.subplots[-1, 2].set_xlabel(_PARAMETER_LABELS["nrun"], fontsize=fontsize)
        plotter.subplots[-1, 3].set_xlabel(_PARAMETER_LABELS[parameter_name], fontsize=fontsize)
        plotter.subplots[2, 0].set_ylabel(_PARAMETER_LABELS["nrun"], fontsize=fontsize)
        plotter.subplots[3, 0].set_ylabel(_PARAMETER_LABELS[parameter_name], fontsize=fontsize)
    elif parameter_name is not None:
        plotter.subplots[-1, 2].set_xlabel(_PARAMETER_LABELS[parameter_name], fontsize=fontsize)
        plotter.subplots[2, 0].set_ylabel(_PARAMETER_LABELS[parameter_name], fontsize=fontsize)

    for index, parameter in enumerate(
        additional_parameters, start=dimension - len(additional_parameters)
    ):
        plotter.subplots[-1, index].set_xlabel(
            _PARAMETER_LABELS[parameter], fontsize=fontsize
        )
        plotter.subplots[index, 0].set_ylabel(
            _PARAMETER_LABELS[parameter], fontsize=fontsize
        )

    for axis in plotter.subplots[-1]:
        for label in axis.get_xticklabels():
            label.set_rotation(45)
            label.set_ha("right")
    if not additional_parameters:
        _set_reference_lines_and_limits(plotter, parameter_name, use_as_ns)


def _print_chain_summaries(
    chains, parameter_name, use_as_ns, additional_parameters
):
    """Print the parameter constraints used when inspecting each figure."""
    for index, samples in enumerate(chains):
        print(f"Sample {index}")
        if use_as_ns:
            print("1 sigma logA", samples.getInlineLatex("logA", limit=1))
            print("1 sigma ns", samples.getInlineLatex("ns", limit=1))

        if parameter_name is None:
            pass
        elif parameter_name == "mnu":
            print(
                "2 sigma mnu",
                samples.getInlineLatex("mnu", limit=2),
            )
        elif parameter_name == "nnu":
            print(
                "1 sigma nnu",
                samples.getInlineLatex("nnu", limit=1, err_sig_figs=3),
            )
        else:
            print(
                f"1 sigma {parameter_name}",
                samples.getInlineLatex(parameter_name, limit=1),
            )
            if parameter_name == "nrunrun":
                print(
                    "1 sigma nrun",
                    samples.getInlineLatex("nrun", limit=1),
                )
        for parameter in additional_parameters:
            if parameter == "mnu":
                print(
                    "2 sigma mnu",
                    samples.getInlineLatex("mnu", limit=2),
                )
            elif parameter == "nnu":
                print(
                    "1 sigma nnu",
                    samples.getInlineLatex("nnu", limit=1, err_sig_figs=3),
                )
            else:
                print(
                    f"1 sigma {parameter}",
                    samples.getInlineLatex(parameter, limit=1),
                )


def _ensure_ranges_periodic(samples):
    """Bridge legacy ``.ranges`` files and current GetDist expectations."""
    if not hasattr(samples.ranges, "periodic"):
        samples.ranges.periodic = set()


def _set_reference_lines_and_limits(plotter, parameter_name, use_as_ns):
    """Add parameter-specific reference lines and historical display limits."""
    if parameter_name == "nnu":
        _add_reference_lines(plotter, 3.046)
        plotter.subplots[-1, 0].set_xlim(0.32, 0.39)
        plotter.subplots[-1, 1].set_xlim(-2.32, -2.275)
        plotter.subplots[-1, -1].set_xlim(3.046 - 0.45, 3.046 + 0.45)
    elif parameter_name == "mnu":
        _add_reference_lines(plotter, 0.06)
        _add_reference_lines(plotter, 0.1)
        plotter.subplots[-1, 0].set_xlim(0.31, 0.385)
        plotter.subplots[-1, 1].set_xlim(-2.31, -2.285)
        plotter.subplots[-1, -1].set_xlim(0.0, 0.35)
    elif parameter_name == "nrun":
        _add_reference_lines(plotter, 0.0)
        plotter.subplots[-1, -1].set_xlim(-0.02, 0.02)
        if use_as_ns:
            plotter.subplots[0, 0].set_xlim(3.005, 3.10)
        else:
            plotter.subplots[0, 0].set_xlim(0.32, 0.38)
            plotter.subplots[1, 1].set_xlim(-2.35, -2.25)
    elif parameter_name == "nrunrun":
        _add_reference_lines(plotter, 0.0, include_running=True)
        plotter.subplots[-2, -2].set_xlim(-0.03, 0.03)
        plotter.subplots[-1, -1].set_xlim(-0.04, 0.04)
        if not use_as_ns:
            plotter.subplots[0, 0].set_xlim(0.305, 0.405)
            plotter.subplots[1, 1].set_xlim(-2.41, -2.17)


def _add_reference_lines(plotter, value, include_running=False):
    """Draw a value in each panel involving the final parameter."""
    plotter.subplots[-1, -1].axvline(value, ls="--", color="black")
    plotter.subplots[-1, 0].axhline(value, ls="--", color="black")
    plotter.subplots[-1, 1].axhline(value, ls="--", color="black")
    if include_running:
        plotter.subplots[-1, 2].axhline(value, ls="--", color="black")
        plotter.subplots[-1, 2].axvline(value, ls="--", color="black")
        plotter.subplots[-2, 0].axhline(value, ls="--", color="black")
        plotter.subplots[-2, 1].axhline(value, ls="--", color="black")
        plotter.subplots[-2, 2].axvline(value, ls="--", color="black")
