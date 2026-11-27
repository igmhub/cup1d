"""Error budget and covariance of the P1D: computed first, then drawn.

p1d_error_budget, plot_error_budget, error_budget_data
    Statistical, systematic, emulator and total errors of a likelihood
    relative to the P1D (Fig. 2 of the DR1 paper).
plot_p1d_correlation
    Correlation matrix of the full P1D covariance of a likelihood.
measurement_error_components, plot_measurement_errors
    Statistical error and every systematic component of a DESI P1D file.
"""

import numpy as np

from cup1d.postprocessing.labels import K_KMS_LABEL
from cup1d.postprocessing.style import styled

ERROR_TERMS = ("stat", "syst", "emu", "total")
_STYLES = {"stat": "-", "syst": ":", "emu": "--", "total": "-."}
_COLOURS = {"stat": "blue", "syst": "orange", "emu": "green", "total": "red"}


def p1d_error_budget(like, key=None, use_pk_smooth=True):
    """Diagonal errors of every redshift bin relative to the P1D.

    Parameters
    ----------
    like : cup1d.likelihood.likelihood.Likelihood
        Likelihood (data covariance with its inflation factors, emulator
        covariance and total covariance).
    key : str, optional
        Data set (default: the first one).
    use_pk_smooth : bool, default=True
        Divide by the smooth P1D instead of the measured one.

    Returns
    -------
    list of dict
        One per redshift: ``z``, ``k`` and the fractional errors ``stat``,
        ``syst`` (both with the configured inflation), ``emu`` and ``total``.
    """
    key = next(iter(like.data)) if key is None else key
    data = like.data[key]
    out = []
    for iz, z in enumerate(data.z):
        stat = np.diag(data.covstat_Pk_kms[iz])
        syst = np.diag(data.cov_Pk_kms[iz]) - stat
        ind = np.argmin(np.abs(like.cov_factor["z"] - z))
        stat = stat * like.cov_factor["val_stat"][ind] ** 2
        syst = syst * like.cov_factor["val_syst"][ind] ** 2
        emu = np.diag(like.cov_emu_Pk_kms[key][iz])
        total = np.diag(like.cov_Pk_kms[key][iz])
        pk = np.asarray(data.Pksmooth_kms[iz] if use_pk_smooth else data.Pk_kms[iz])
        out.append({"z": float(z), "k": np.asarray(data.k_kms[iz]),
                    **{name: np.sqrt(var) / pk for name, var in
                       zip(ERROR_TERMS, (stat, syst, emu, total))}})
    return out


def error_budget_data(budget):
    """Arrays of :func:`plot_error_budget` (``x<i>``, ``y<i>_<colour>`` as in the Zenodo files)."""
    out = {}
    for i, entry in enumerate(budget):
        out[f"x{i}"] = entry["k"]
        for name in ERROR_TERMS:
            out[f"y{i}_{_COLOURS[name]}"] = entry[name]
    return out


@styled
def plot_error_budget(budget, fontsize=18, ylims=((0.0, 0.06), (0.0, 0.06))):
    """Fractional errors against k, one panel per redshift.

    Parameters
    ----------
    budget : list of dict
        Output of :func:`p1d_error_budget`.
    fontsize : float, default=18
        Base font size.
    ylims : sequence of (float, float), optional
        Y limits of the first two panel rows.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    nz = len(budget)
    ncols = int(np.round(np.sqrt(nz)))
    fig, axes = plt.subplots(ncols + 1, ncols, sharex=True, sharey="row", figsize=(10, 8))
    axes = axes.reshape(-1)
    for ax, entry in zip(axes, budget):
        for name in ERROR_TERMS:
            ax.plot(entry["k"], entry[name], ls=_STYLES[name], lw=3)
        ax.text(0.05, 0.95, f"z={entry['z']:g}", ha="left", va="top", transform=ax.transAxes,
                fontsize=fontsize)
        ax.tick_params(axis="both", which="major", labelsize=fontsize)
    for ax in axes[nz:]:
        ax.axis("off")
    for name in ERROR_TERMS:
        axes[-1].plot([0, 0], [0, 0], ls=_STYLES[name], lw=3,
                      label=r"$\sigma_x = \sigma_\mathrm{" + name + "}$")
    axes[-1].legend(fontsize=fontsize, loc="upper left")
    fig.supxlabel(K_KMS_LABEL, fontsize=fontsize + 2)
    fig.supylabel(r"$\sigma_x/P_\mathrm{1D}$", fontsize=fontsize + 2)
    if ylims is not None:
        for row, limits in enumerate(ylims):
            if row * ncols < nz:
                axes[row * ncols].set_ylim(limits)
    fig.tight_layout()
    return fig


@styled
def plot_p1d_correlation(like, key=None, fontsize=12):
    """Correlation matrix of the full P1D covariance of a likelihood.

    Parameters
    ----------
    like : cup1d.likelihood.likelihood.Likelihood
        Likelihood with ``full_cov_Pk_kms`` (per data set, or one matrix).
    key : str, optional
        Data set (default: the first one).
    fontsize : float, default=12
        Font size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    from cup1d.postprocessing.primitives import correlation_from_covariance, draw_correlation_matrix

    covariance = like.full_cov_Pk_kms
    if isinstance(covariance, dict):
        covariance = covariance[next(iter(covariance)) if key is None else key]
    fig, ax = plt.subplots(figsize=(8, 7))
    draw_correlation_matrix(ax, correlation_from_covariance(covariance), lower=False,
                            cmap="viridis", vmin=None, vmax=None, colorbar="side",
                            fontsize=fontsize)
    ax.set_xlabel("data index", fontsize=fontsize)
    ax.set_ylabel("data index", fontsize=fontsize)
    fig.tight_layout()
    return fig


SYSTEMATICS = {
    "QMLE": ("E_DLA_COMPLETENESS", "E_BAL_COMPLETENESS", "E_RESOLUTION", "E_CONTINUUM",
             "E_CONTINUUM_ADD", "E_NOISE_SCALE", "E_NOISE_ADD"),
    "FFT": ("E_PSF", "E_RESOLUTION", "E_SIDE_BAND", "E_LINES", "E_DLA", "E_BAL", "E_CONTINUUM",
            "E_DLA_COMPLETENESS", "E_BAL_COMPLETENESS"),
}


def measurement_error_components(p1d_fname, kmin=1e-3, nknyq=0.5):
    """Statistical error and systematic components of a DESI P1D file, relative to the P1D.

    Parameters
    ----------
    p1d_fname : str or path-like
        DESI P1D FITS file (``P1D_BLIND``, ``COVARIANCE_STAT`` and
        ``SYSTEMATICS`` extensions) whose name contains ``fft`` or ``qmle``.
    kmin : float, default=1e-3
        Smallest wavenumber kept, s/km.
    nknyq : float, default=0.5
        Largest wavenumber kept, as a fraction of the Nyquist wavenumber of
        the redshift.

    Returns
    -------
    dict
        ``measurement`` ("QMLE" or "FFT") and ``bins``: per redshift ``z``,
        ``k`` and ``{component: sigma/P}`` (``stat`` and the systematics).

    Raises
    ------
    ValueError
        If the measurement type cannot be read from the file name.
    """
    from astropy.io import fits

    name = str(p1d_fname)
    if "fft" in name:
        measurement = "FFT"
    elif "qmle" in name:
        measurement = "QMLE"
    else:
        raise ValueError(f"cannot tell the measurement type (fft or qmle) of {name}")
    with fits.open(name) as hdu:
        extensions = {h.header["EXTNAME"]: h for h in hdu if "EXTNAME" in h.header}
        pk = np.asarray(extensions["P1D_BLIND"].data["PLYA"])
        stat = np.sqrt(np.diag(np.asarray(extensions["COVARIANCE_STAT"].data)))
        syst = extensions["SYSTEMATICS"].data
        zs = np.asarray(syst["Z"])
        ks = np.asarray(syst["K"])
        components = {label: np.asarray(syst[label]) for label in SYSTEMATICS[measurement]}
    bins = []
    for z in np.unique(zs):
        k_nyq = np.pi / (2.99792458e5 * 0.8 / 1215.67 / (1 + z))
        ind = np.flatnonzero((zs == z) & (ks > kmin) & (ks < k_nyq * nknyq))
        errors = {"stat": stat[ind] / pk[ind]}
        errors.update({label: values[ind] / pk[ind] for label, values in components.items()})
        bins.append({"z": float(z), "k": ks[ind], "errors": errors})
    return {"measurement": measurement, "bins": bins}


@styled
def plot_measurement_errors(components, fontsize=14):
    """Every error component of a DESI P1D file against k, one panel per redshift.

    Parameters
    ----------
    components : mapping
        Output of :func:`measurement_error_components`.
    fontsize : float, default=14
        Font size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    bins = components["bins"]
    nrows = int(np.ceil(len(bins) / 3))
    fig, axes = plt.subplots(nrows, 3, sharex=True, sharey=True, figsize=(20, 16))
    axes = axes.reshape(-1)
    for ax, entry in zip(axes, bins):
        for i, (label, values) in enumerate(entry["errors"].items()):
            if label == "stat":
                ax.plot(entry["k"], values, color="k", label=label)
                continue
            keep = np.isfinite(values) & (values > 0)
            ax.plot(entry["k"][keep], values[keep], color=f"C{i - 1}", label=label)
        ax.set_title(f"$z = ${entry['z']:g}", fontsize=fontsize)
    for ax in axes[len(bins):]:
        ax.axis("off")
    axes[0].set_yscale("log")
    axes[0].set_xscale("log")
    axes[0].set_ylim(1e-3, 0.5)
    axes[0].legend(fontsize=fontsize - 4, loc="upper right", ncol=3)
    fig.supxlabel(K_KMS_LABEL, fontsize=fontsize)
    fig.supylabel(r"$\sigma/P$", fontsize=fontsize)
    fig.suptitle(components["measurement"], fontsize=fontsize + 2)
    fig.tight_layout()
    return fig
