import os

from lace.emulator import emulator_manager
from cup1d.pipeline import set_archive


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
            archive = set_archive(training_set)
    else:
        archive = None
    #######################

    if emulator_label == "forest_mpg":
        old_emu = False
        if old_emu:
            # old emu, better accuracy
            from forestflow.old_emu.paper_P3D_cINN import P3DEmulator as old_P3DEmulator
            from forestflow.archive import GadgetArchive3D

            Archive3D = GadgetArchive3D()
            emulator = old_P3DEmulator(
                Archive3D.training_data,
                Archive3D.emu_params,
                nLayers_inn=12,
                Archive=Archive3D,
                Nrealizations=3000,
                training_type="Arinyo_min",
                model_path=os.path.join(
                    os.path.dirname(forestflow.__path__[0]),
                    "data",
                    "emulator_models",
                    "mpg_hypercube.pt",
                ),
            )
        else:
            # new emu, worse accuracy
            import forestflow
            from forestflow.P3D_cINN import P3DEmulator

            emulator = P3DEmulator(
                model_path=os.path.join(
                    os.path.dirname(forestflow.__path__[0]),
                    "data",
                    "emulator_models",
                    "forest_mpg",
                )
            )

        # TBD add within forestflow
        # compute l10 error from forestflow
        emulator.emulator_label = "forest_mpg"
        emulator.kp_Mpc = 0.7
        list_sim_cube = []
        for ii in range(30):
            list_sim_cube.append("mpg" + str(ii))

        emulator.list_sim_cube = list_sim_cube

    else:
        emulator = emulator_manager.set_emulator(
            emulator_label=emulator_label,
            archive=archive,
            drop_sim=drop_sim,
        )

    return emulator
