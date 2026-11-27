from lace.cosmo.cosmology import Cosmology
from lace.cosmo.labeled_cosmologies import (
    COSMOLOGY_LABELS,
    DEFAULT_NYX_VERSION,
    DEPRECATED_LABELS,
    get_cosmo_params_dict_from_label,
    get_sim_suite_cosmologies,
)


def set_cosmo(
    cosmo_label="mpg_central",
    return_all=False,
    nyx_version=DEFAULT_NYX_VERSION,
):
    """Construct a fiducial cosmology from a named simulation or reference.

    The labels are defined in LaCE: see
    :func:`lace.cosmo.labeled_cosmologies.get_cosmo_params_dict_from_label`.

    Parameters
    ----------
    cosmo_label : str, default: "mpg_central"
        Simulation (``mpg_*``, ``nyx_*``, ``accel2``, ``sherwood``), Planck,
        DESI, or reference-cosmology label.
    return_all : bool, default: False
        For MPG and Nyx catalogue labels, return the complete stored catalogue
        rather than the selected cosmology.
    nyx_version : str, default: "models_Nyx_Mar2025_with_CGAN_val_3axes"
        Nyx emulator catalogue version used for Nyx labels.

    Returns
    -------
    lace.cosmo.cosmology.Cosmology or dict
        Selected cosmology, or the selected simulation-suite catalogue when
        ``return_all`` is true for a catalogue-backed label.

    Raises
    ------
    ValueError
        If an input catalogue cannot be loaded or the label is unknown.
    """
    if cosmo_label in COSMOLOGY_LABELS or cosmo_label in DEPRECATED_LABELS:
        cosmo = Cosmology(cosmo_label=cosmo_label)
    else:
        cosmo = Cosmology(
            cosmo_params_dict=get_cosmo_params_dict_from_label(
                cosmo_label, nyx_version=nyx_version
            )
        )

    if return_all:
        if cosmo_label[:3] not in ("mpg", "nyx"):
            raise ValueError("return_all requires an mpg_* or nyx_* cosmo_label")
        return get_sim_suite_cosmologies(cosmo_label[:3], nyx_version=nyx_version)
    return cosmo
