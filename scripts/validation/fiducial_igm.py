import os

from cup1d.likelihood.input_pipeline import Args
from lace.emulator.emulator_manager import set_emulator
from cup1d.likelihood.pipeline import set_archive, Pipeline


def main():
    """Launch validate cosmology"""

    # emulator_label = "Pedersen23_ext"
    # emulator_label = "Cabayol23+"
    # training_set = "Cabayol23"
    # suite = "mpg"

    emulator_label = "Nyx_alphap_cov"
    training_set = "Nyx23_Jul2024"
    suite = "nyx"
    nIGM = 2

    base_out_folder = (
        "/home/jchaves/Proyectos/projects/lya/data/cup1d/validate_igm/"
    )

    validate_cosmo(emulator_label, training_set, base_out_folder, suite, nIGM)


def validate_cosmo(emulator_label, training_set, base_out_folder, suite, nIGM):
    """Validate assuming a fiducial cosmology against forecast data"""

    args = Args(emulator_label=emulator_label, training_set=training_set)

    # set true igm
    args.true_igm_label = suite + "_central"

    # set covariance matrix
    args.data_label = "mock_DESIY1"
    # args.data_label = "mock_Chabanier2019"
    args.p1d_fname = "/home/jchaves/Proyectos/projects/lya/data/cup1d/obs/desi_y1_baseline_p1d_sb1subt_qmle_power_estimate.fits"

    # set number of free IGM parameters
    args.n_tau = nIGM
    args.n_sigT = nIGM
    args.n_gamma = nIGM
    args.n_kF = nIGM

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

    # same true and fiducial cosmology
    if suite == "nyx":
        args.true_cosmo_label = "nyx_central"
        args.fid_cosmo_label = "nyx_central"
    else:
        args.true_cosmo_label = "mpg_central"
        args.fid_cosmo_label = "mpg_central"

    for sim_label in args.emulator.list_sim_cube:
        print("\n\n\n")
        print(sim_label)
        print("\n\n\n")

        # set fiducial igm (the only thing that changes)
        args.fid_igm_label = sim_label

        out_folder = os.path.join(
            base_out_folder,
            args.data_label[5:],
            emulator_label,
            "nIGM" + str(nIGM),
            sim_label,
        )

        pip = Pipeline(args, make_plots=False, out_folder=out_folder)
        pip.run_minimizer()


if __name__ == "__main__":
    main()
