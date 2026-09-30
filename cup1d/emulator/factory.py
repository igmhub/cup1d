def set_emulator(emulator_label="CH24_mpgcen_gpr"):
    """Load a supported LaCE or legacy ForestFlow emulator."""

    if emulator_label == "forest_mpg":
        from forestflow.emulator import P1DEmulator

        emulator = P1DEmulator()
    else:
        from lace.emulator import emulator_manager

        emulator = emulator_manager.set_emulator(
            emulator_label=emulator_label,
        )

    return emulator
