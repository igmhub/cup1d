"""Products written next to a native cup1d (emcee) chain, besides the figures.

``line_sigmas.npy`` / ``line_sigmas_Asns.npy``
    68 and 95 per cent contour polygons of (Delta2_star, n_star) and (As, ns),
    read by the comparison figures (variations, historical constraints,
    Planck contours).
``summary.npy``
    chi2 and probability of the best fit and star-parameter summaries, read
    by the variation tables.

The figures themselves are written by
:class:`cup1d.postprocessing.posterior.PosteriorPlotter`.
"""

import os

import numpy as np

CONTOUR_LEVELS = (0.68, 0.95)


def load_parameter_bounds(folder):
    """Return free-parameter names and prior bounds of a chain folder.

    Reads the ``sampler_results.npy`` manifest written by
    :meth:`cup1d.inference.fitter.Fitter.save_sampler_results`, falling back to
    the legacy ``fitter_results.npy`` product of older runs.

    Returns
    -------
    names : list of str
        Free-parameter names in chain order.
    bounds : dict
        ``{name: (min_value, max_value)}`` prior bounds.

    Raises
    ------
    FileNotFoundError
        If neither product exists.
    KeyError
        If the manifest predates the stored ``parameter_bounds``.
    """
    manifest = os.path.join(folder, "sampler_results.npy")
    if os.path.isfile(manifest):
        payload = np.load(manifest, allow_pickle=True).item()
        if "parameter_bounds" not in payload:
            raise KeyError(
                f"{manifest} has no parameter_bounds; it was written before they "
                "were stored. Re-save it or plot from the legacy products."
            )
        names = list(payload["parameter_names"])
        return names, {name: tuple(payload["parameter_bounds"][name]) for name in names}

    legacy = os.path.join(folder, "fitter_results.npy")
    if not os.path.isfile(legacy):
        raise FileNotFoundError(f"Neither sampler_results.npy nor fitter_results.npy in {folder}")
    fdict = np.load(legacy, allow_pickle=True).item()
    names = list(fdict["like"]["free_param_names"])
    free_params = fdict["like"]["free_params"]
    return names, {name: (free_params[name]["min_value"], free_params[name]["max_value"])
                   for name in names}


def contour_polygons(x, y, weights=None, levels=CONTOUR_LEVELS, bins=50, smooth=1.0):
    """Highest-density contours of 2D samples as polygons.

    Returns
    -------
    dict
        ``{level: [(x_outline, y_outline), ...]}``, the format of
        ``line_sigmas.npy`` (and of
        :func:`cup1d.postprocessing.primitives.draw_contour_polygons`).
    """
    import contourpy

    from cup1d.postprocessing.posterior.stats import density_contours

    X, Y, H, thresholds = density_contours(x, y, weights, levels=levels, bins=bins, smooth=smooth)
    generator = contourpy.contour_generator(X, Y, H)
    # thresholds are increasing: widest level first
    by_level = dict(zip(sorted(levels, reverse=True), thresholds[:-1]))
    return {level: [(line[:, 0], line[:, 1]) for line in generator.lines(by_level[level])]
            for level in levels}


def save_contours(x, y, folder, flag="", weights=None):
    """Write :func:`contour_polygons` to ``<folder>/line_sigmas<flag>.npy``."""
    path = os.path.join(folder, f"line_sigmas{flag}.npy")
    np.save(path, contour_polygons(x, y, weights))
    return path


def chain_summary(folder, samples, lnprob_max=None):
    """Summary of a chain: best-fit chi2 and probability, star parameters.

    Parameters
    ----------
    folder : str or path-like
        Chain folder (``sampler_results.npy`` or legacy ``fitter_results.npy``).
    samples : PosteriorSamples
        Samples of the chain (star parameters as stored, i.e. blinded).
    lnprob_max : float, optional
        Highest log probability of the chain; compared with the saved best fit.

    Returns
    -------
    dict
        ``ndata``, ``npar``, ``ndeg``, ``chi2``, ``prob_chi2``, the centre of
        the 68 per cent (Delta2_star, n_star) contour, and the 16/50/84
        quantiles and half-widths of both.
    """
    from scipy.stats import chi2 as chi2_distribution

    out = {}
    manifest = os.path.join(folder, "sampler_results.npy")
    if os.path.isfile(manifest):
        data = np.load(manifest, allow_pickle=True).item()
        out["ndata"] = data["n_data"]
        out["npar"] = len(data["parameter_names"])
        lnprob_mle = -0.5 * data["fit"]["mle_chi2"]
    else:
        data = np.load(os.path.join(folder, "fitter_results.npy"), allow_pickle=True).item()
        if "full_Pk_kms" in data["data"]:
            out["ndata"] = data["data"]["full_Pk_kms"].shape[0]
        else:
            out["ndata"] = int(sum(np.shape(pk)[0] for pk in data["data"]["Pk_kms"]))
        out["npar"] = data["fitter"]["mle_cube"].shape[0]
        lnprob_mle = data["fitter"]["lnprob_mle"]
    out["ndeg"] = out["ndata"] - out["npar"]
    best = lnprob_mle if lnprob_max is None else max(lnprob_mle, lnprob_max)
    out["chi2"] = -2 * best
    out["prob_chi2"] = chi2_distribution.sf(out["chi2"], out["ndeg"])

    contours = contour_polygons(samples.column("Delta2_star"), samples.column("n_star"), samples.weights)
    inner = contours[CONTOUR_LEVELS[0]][0]
    out["delta2_star_2dcen"] = np.median(inner[0])
    out["nstar_2dcen"] = np.median(inner[1])
    for name, key in (("Delta2_star", "delta2_star"), ("n_star", "n_star")):
        q = samples.quantiles(name, (0.16, 0.5, 0.84))
        out[f"{key}_16_50_84"] = q
        out[f"{key}_err"] = 0.5 * (q[2] - q[0])
    return out


def write_chain_products(folder, out_folder=None, nburn_extra=0, thin=1):
    """Write ``line_sigmas.npy``, ``line_sigmas_Asns.npy`` and ``summary.npy``.

    Parameters
    ----------
    folder : str or path-like
        Chain folder (``chain.npy``, ``blobs.npy``, ``lnprob.npy`` and the
        sampler manifest).
    out_folder : str or path-like, optional
        Output folder (default: ``folder``).
    nburn_extra : int, default=0
        Extra initial steps to discard.
    thin : int, default=1
        Keep every ``thin``-th step.

    Returns
    -------
    dict
        The summary written to ``summary.npy``.
    """
    from cup1d.postprocessing.posterior.samples import PosteriorSamples

    out_folder = folder if out_folder is None else out_folder
    os.makedirs(out_folder, exist_ok=True)
    samples = PosteriorSamples.from_cup1d_folder(folder, nburn_extra=nburn_extra, thin=thin)
    save_contours(samples.column("Delta2_star"), samples.column("n_star"), out_folder)
    if "As" in samples and "ns" in samples:
        save_contours(samples.column("As"), samples.column("ns"), out_folder, flag="_Asns")
    summary = chain_summary(folder, samples, lnprob_max=-float(np.min(samples.minuslogpost)))
    np.save(os.path.join(out_folder, "summary.npy"), summary)
    return summary
