from cup1d.likelihood.input_pipeline import Args
from lace.emulator.emulator_manager import set_emulator
from cup1d.likelihood.pipeline import set_archive, Pipeline


def main():
    """Launch validate cosmology"""

    emulator_label = "Pedersen23_ext"
    # emulator_label = "Cabayol23+"
    training_set = "Cabayol23"

    # emulator_label = "Cabayol23+"
    # training_set = "Cabayol23"
    base_out_folder = (
        "/home/jchaves/Proyectos/projects/lya/data/cup1d/validate_cosmo/"
    )

    validate_cosmo(emulator_label, training_set, base_out_folder)


def validate_cosmo(emulator_label, training_set, base_out_folder):
    """Validate assuming a fiducial cosmology against forecast data"""

    args = Args(emulator_label=emulator_label, training_set=training_set)

    # set fiducial cosmology
    args.fid_cosmo_label = "Planck18"

    # set covariance matrix
    # args.data_label = "mock_DESIY1"
    args.data_label = "mock_Chabanier2019"

    # set number of free IGM parameters
    args.n_tau = 2
    args.n_sigT = 2
    args.n_gamma = 2
    args.n_kF = 2

    # set archive and emulator
    args.archive = set_archive(args.training_set)
    args.emulator = set_emulator(
        emulator_label=args.emulator_label,
        archive=args.archive,
    )

    if "Nyx" in args.emulator.emulator_label:
        args.emulator.list_sim_cube = args.archive.list_sim_cube
        args.vary_alphas = True
        if "nyx_14" in args.emulator.list_sim_cube:
            args.emulator.list_sim_cube.remove("nyx_14")
    else:
        args.emulator.list_sim_cube = args.archive.list_sim_cube
        args.vary_alphas = False

    # same true and fiducial IGM
    if "Nyx" in args.emulator.emulator_label:
        args.true_igm_label = "nyx_central"
        args.fid_igm_label = "nyx_central"
    else:
        args.true_igm_label = "mpg_central"
        args.fid_igm_label = "mpg_central"

    # loop over simulations
    for sim_label in args.emulator.list_sim_cube:
        print("\n\n\n")
        print(sim_label)
        print("\n\n\n")

        # set fiducial cosmology (the only thing that changes)
        args.fid_cosmo_label = sim_label

        out_folder = (
            base_out_folder + "/" + emulator_label + "/" + sim_label + "/"
        )

        pip = Pipeline(args, make_plots=False, out_folder=out_folder)
        pip.run_minimizer()


if __name__ == "__main__":
    main()