def set_emulator(
    emulator_label="CH24_mpgcen_gpr",
    drop_sim=None,
    archive=None,
    training_set="Cabayol23",
):
    """
    Set emulator
    """

    # only read archive if using old emulator
    if emulator_label not in [
        "CH24_mpg_gp",
        "CH24_nyx_gp",
        "CH24_mpgcen_gpr",
        "CH24_nyxcen_gpr",
        "forest_mpg",
    ]:
        read_archive = True
    else:
        read_archive = False

    if read_archive:
        if archive is None:
            from cup1d.pipeline import set_archive

            archive = set_archive(training_set)
    else:
        archive = None
    #######################

    if emulator_label == "forest_mpg":
        from cup1d.likelihood.interface_emu import P1D_emulator

        emulator = P1D_emulator()
    else:
        from lace.emulator import emulator_manager

        emulator = emulator_manager.set_emulator(
            emulator_label=emulator_label,
            archive=archive,
            drop_sim=drop_sim,
        )

    return emulator
