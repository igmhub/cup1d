"""Factory helpers for Lyman-alpha P1D emulators."""

from lace.emulator import emulator_manager

from cup1d.pipeline.set_archive import set_archive


def set_emulator(
    emulator_label="CH24_mpgcen_gpr",
    drop_sim=None,
    archive=None,
    training_set="Cabayol23",
):
    """Build an emulator from its label.

    Parameters
    ----------
    emulator_label : str, optional
        Name understood by :mod:`lace.emulator.emulator_manager`.
    drop_sim : str or list[str] or None, optional
        Simulation(s) to omit when constructing archive-backed emulators.
    archive : object or None, optional
        Preloaded simulation archive. If omitted, older emulator labels load
        an archive using ``training_set``.
    training_set : str, optional
        Archive training-set label used for older emulator configurations.

    Returns
    -------
    object
        Configured emulator instance.
    """

    # only read archive if using old emulator
    if emulator_label not in [
        "CH24_mpg_gp",
        "CH24_nyx_gp",
        "CH24_mpgcen_gpr",
        "CH24_nyxcen_gpr",
    ]:
        read_archive = True
    else:
        read_archive = False

    if read_archive:
        if archive is None:
            archive = set_archive(training_set=training_set)
    else:
        archive = None
    #######################

    emulator = emulator_manager.set_emulator(
        emulator_label=emulator_label,
        archive=archive,
        drop_sim=drop_sim,
    )

    return emulator
