from lace.emulator import emulator_manager


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
    ]:
        read_archive = True
    else:
        read_archive = False

    if read_archive:
        if archive is None:
            archive = set_archive(args.training_set)
    else:
        archive = None
    #######################

    emulator = emulator_manager.set_emulator(
        emulator_label=emulator_label,
        archive=archive,
        drop_sim=drop_sim,
    )

    return emulator
