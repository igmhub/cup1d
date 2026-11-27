"""External constraints and the DESI DR1 products the figures compare with.

LITERATURE
    Published Gaussian constraints on (Delta2_star, n_star), by name:
    the function of :mod:`cup1d.likelihood.marginal` that evaluates them, the
    legend label, colour and line style.
literature_igm
    Literature IGM measurements (Gaikwad+21, Turner+24).
dr1_output_root, dr1_chain_directory, load_blinding_dr1, load_dr1_star_samples
    The DESI DR1 baseline chain (Chaves-Montero+26) and its blinding offsets.
"""

from pathlib import Path

import numpy as np

# name -> (function of cup1d.likelihood.marginal, legend label, colour, line style)
LITERATURE = {
    "McDonald2005": ("gaussian_chi2_McDonald2005", "SDSS (McDonald+05)", "C1", ":"),
    "PalanqueDelabrouille2015": ("gaussian_chi2_PalanqueDelabrouille2015",
                                 "BOSS (Palanque-Delabrouille+15)", "C2", "-."),
    "Chabanier2019": ("gaussian_chi2_Chabanier2019", "eBOSS (Chabanier+19)", "C3", "--"),
    "Walther2024": ("gaussian_chi2_Walther2024", "eBOSS + priors (Walther+24)", "C4", "--"),
}
DR1 = ("gaussian_chi2_DESI_DR1", "DESI DR1 (Chaves-Montero+26)", "k", "-")


def gaussian_constraint(function_name):
    """Return the Gaussian summary of a published compressed constraint.

    Parameters
    ----------
    function_name : str
        Function of :mod:`cup1d.likelihood.marginal`, e.g.
        ``"gaussian_chi2_DESI_DR1"``.

    Returns
    -------
    dict
        ``Delta2_star``, ``Delta2_star_err``, ``n_star``, ``n_star_err`` and
        the correlation ``r``.
    """
    from cup1d.likelihood import marginal

    result = getattr(marginal, function_name)(-2.3, 0.35)
    return {key: result[key] for key in ("Delta2_star", "Delta2_star_err", "n_star", "n_star_err", "r")}


def gaussian_chi2(function_name, n_star, delta2_star):
    """Evaluate a published Gaussian constraint (chi2) on a grid."""
    from cup1d.likelihood import marginal

    function = getattr(marginal, function_name)
    if function_name == "gaussian_chi2_Walther2024":
        return function(n_star, delta2_star, ana_type="priors")["chi2"]
    return function(n_star, delta2_star)["chi2"]


def literature_igm():
    """Literature IGM measurements in cup1d conventions.

    Returns
    -------
    dict
        ``{"Gaikwad+21": {...}, "Turner+24": {...}}``, each with ``z`` and
        quantities (``mF``, ``tau_eff``, ``T0`` in 10^4 K, ``sigT_kms``,
        ``gamma``) with their ``<quantity>_err``.
    """
    # the table lives with the IGM model, which also uses it for priors
    from cup1d.likelihood.likelihood import others_igm

    return dict(zip(("Gaikwad+21", "Turner+24"), others_igm()))


def dr1_output_root():
    """Folder of the DESI DR1 runs: ``<cup1d>/../../data/out_DESI_DR1``."""
    from cup1d.utils.utils import get_path_repo

    return Path(get_path_repo("cup1d")).parents[1] / "data" / "out_DESI_DR1"


def dr1_chain_directory(output_root=None):
    """Folder of the DESI DR1 baseline chain (``.../global_opt/CH24_mpgcen_gpr/chain_7``)."""
    root = Path(output_root) if output_root is not None else dr1_output_root()
    return root / "DESIY1_QMLE3" / "global_opt" / "CH24_mpgcen_gpr" / "chain_7"


def load_blinding_dr1():
    """Additive DR1 blinding offsets ``{"Delta2_star": ..., "n_star": ...}``."""
    from cup1d.utils.utils import get_path_repo

    path = Path(get_path_repo("cup1d")) / "data" / "blinding_dr1.npy"
    return np.load(path, allow_pickle=True).item()


def load_dr1_star_samples(chain_directory=None, unblind=True):
    """Delta2_star and n_star samples of a DR1 chain (``blobs.npy``).

    Parameters
    ----------
    chain_directory : str or path-like, optional
        Chain folder; :func:`dr1_chain_directory` by default.
    unblind : bool, default=True
        Subtract the DR1 blinding offsets (:func:`load_blinding_dr1`).

    Returns
    -------
    dict
        ``{"Delta2_star": array, "n_star": array}`` (flattened).
    """
    from cup1d.utils.various_dicts import get_blob_value

    folder = Path(chain_directory) if chain_directory is not None else dr1_chain_directory()
    blobs = np.load(folder / "blobs.npy", mmap_mode="r")
    offsets = load_blinding_dr1() if unblind else {}
    return {name: np.asarray(get_blob_value(blobs, name)).reshape(-1) - offsets.get(name, 0.0)
            for name in ("Delta2_star", "n_star")}
