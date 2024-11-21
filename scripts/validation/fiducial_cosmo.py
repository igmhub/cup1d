import os

from cup1d.likelihood.input_pipeline import Args
from lace.emulator.emulator_manager import set_emulator
from cup1d.likelihood.pipeline import set_archive, Pipeline


def main():
    """Launch validate cosmology"""

    # emulator_label = "Pedersen23_ext"
    # emulator_label = "Cabayol23+"
    # training_set = "Cabayol23"
    emulator_label = "Nyx_alphap_cov"
    training_set = "Nyx23_Jul2024"
    nIGM = 0

    base_out_folder = (
        "/home/jchaves/Proyectos/projects/lya/data/cup1d/validate_cosmo/"
    )

    validate_cosmo(emulator_label, training_set, base_out_folder, nIGM)


def validate_cosmo(emulator_label, training_set, base_out_folder, nIGM):
    """Validate assuming a fiducial cosmology against forecast data"""

    args = Args(emulator_label=emulator_label, training_set=training_set)

    # set true cosmology
    args.true_cosmo_label = "Planck18"

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
        sim_label = []
        for ii in range(14):
            sim_label.append("nyx_" + str(ii))
    else:
        args.emulator.list_sim_cube = args.archive.list_sim_cube
        args.vary_alphas = False
        list_sims = args.emulator.list_sim_cube

    for sim_label in list_sims:
        print("\n\n\n")
        print(sim_label)
        print("\n\n\n")

        # set fiducial cosmology (the only thing that changes)
        args.fid_cosmo_label = sim_label
        # same true and fiducial IGM
        args.true_igm_label = sim_label
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
        # break


if __name__ == "__main__":
    main()
