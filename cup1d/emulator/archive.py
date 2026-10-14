from lace.archive import gadget_archive, nyx_archive


def set_archive(training_set="Pedersen21"):
    """Construct the simulation archive for a supported training set.

    Parameters
    ----------
    training_set : str, default="Pedersen21"
        ``"Pedersen21"`` or ``"Cabayol23"`` selects an MP-Gadget archive.
        Any label containing ``"Nyx"`` selects a Nyx archive of that version.

    Returns
    -------
    lace.archive.gadget_archive.GadgetArchive or lace.archive.nyx_archive.NyxArchive
        Archive selected by ``training_set``.

    Raises
    ------
    UnboundLocalError
        If the label is neither an MPG set nor contains ``"Nyx"``. Callers
        should use :func:`cup1d.configuration.args.get_training_set`.
    """
    if "Nyx" in training_set:
        archive = nyx_archive.NyxArchive(nyx_version=training_set)
    elif training_set in ["Pedersen21", "Cabayol23"]:
        archive = gadget_archive.GadgetArchive(postproc=training_set)
    return archive
