_EMULATOR_ALIASES = {
    "lace_mpg": "CH24_mpgcen_gpr",
    "lace_nyx": "CH24_nyxcen_gpr",
}


def set_emulator(emulator_label="lace_mpg"):
    """Load a LaCE or ForestFlow emulator after resolving cup1d aliases.

    Parameters
    ----------
    emulator_label : str, default="lace_mpg"
        ``lace_mpg`` and ``lace_nyx`` resolve to LaCE central GP bundles;
        ``forest_mpg`` loads ForestFlow's default (current) MPG bundle.

    Returns
    -------
    object
        LaCE's configured GP emulator or ForestFlow's :class:`P1DEmulator`.
    """

    emulator_label = _EMULATOR_ALIASES.get(emulator_label, emulator_label)

    if emulator_label == "forest_mpg":
        from forestflow.emulator import P1DEmulator

        # ForestFlow decides which of its bundles is the current one.
        emulator = P1DEmulator()
    else:
        from lace.emulator import emulator_manager

        emulator = emulator_manager.set_emulator(
            emulator_label=emulator_label,
        )

    return emulator
