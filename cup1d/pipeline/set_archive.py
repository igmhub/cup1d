from lace.archive import gadget_archive, nyx_archive


def set_archive(training_set="Pedersen21"):
    """Set archive

    Parameters
    ----------
    training_set : str

    Returns
    -------
    archive : object

    """
    if "Nyx" in training_set:
        archive = nyx_archive.NyxArchive(nyx_version=training_set)
    elif training_set in ["Pedersen21", "Cabayol23"]:
        archive = gadget_archive.GadgetArchive(postproc=training_set)
    return archive
