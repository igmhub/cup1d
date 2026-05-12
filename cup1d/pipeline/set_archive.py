"""Archive factory helpers for emulator training sets."""

from lace.archive import gadget_archive, nyx_archive


def set_archive(training_set="Pedersen21"):
    """Return the simulation archive for a named training set.

    Parameters
    ----------
    training_set : str, optional
        Training-set identifier. Nyx labels are passed to
        :class:`lace.archive.nyx_archive.NyxArchive`; supported Gadget labels
        are ``"Pedersen21"`` and ``"Cabayol23"``.

    Returns
    -------
    object
        Configured archive instance.
    """
    if "Nyx" in training_set:
        archive = nyx_archive.NyxArchive(nyx_version=training_set)
    elif training_set in ["Pedersen21", "Cabayol23"]:
        archive = gadget_archive.GadgetArchive(postproc=training_set)
    else:
        raise ValueError(f"training_set {training_set} not implemented")
    return archive
