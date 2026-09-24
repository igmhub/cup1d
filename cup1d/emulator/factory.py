def set_emulator(
    emulator_label="CH24_mpgcen_gpr",
    drop_emu_sim=None,
    archive=None,
    training_set="Cabayol23",
):
    """Load a supported LaCE or legacy ForestFlow emulator.

    ``archive``, ``training_set``, and ``drop_emu_sim`` are retained only for
    backwards-compatible calls; current emulator implementations do not use them.
    """

    if emulator_label == "forest_mpg":
        from cup1d.emulator.interface import P1D_emulator

        emulator = P1D_emulator()
    else:
        from lace.emulator import emulator_manager

        emulator = emulator_manager.set_emulator(
            emulator_label=emulator_label,
        )

    return emulator
