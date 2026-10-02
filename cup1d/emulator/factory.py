_EMULATOR_ALIASES = {
    "lace_mpg": "CH24_mpgcen_gpr",
    "lace_nyx": "CH24_nyxcen_gpr",
    "forest_mpg": "forest_mpg_fix",
    "forest_mpg_old": "forest_mpg",
}


def set_emulator(emulator_label="lace_mpg"):
    """Load a LaCE or ForestFlow emulator after resolving cup1d aliases.

    Parameters
    ----------
    emulator_label : str, default="lace_mpg"
        ``lace_mpg`` and ``lace_nyx`` resolve to LaCE central GP bundles;
        ``forest_mpg`` resolves to the corrected ForestFlow MPG bundle;
        ``forest_mpg_old`` resolves through the legacy ForestFlow name.

    Returns
    -------
    object
        LaCE's configured GP emulator or ForestFlow's :class:`P1DEmulator`.
    """

    emulator_label = _EMULATOR_ALIASES.get(emulator_label, emulator_label)

    if emulator_label in ("forest_mpg", "forest_mpg_fix"):
        from forestflow.emulator import P1DEmulator

        emulator = P1DEmulator(name_emu=emulator_label)
    else:
        from lace.emulator import emulator_manager

        emulator = emulator_manager.set_emulator(
            emulator_label=emulator_label,
        )

    return emulator
