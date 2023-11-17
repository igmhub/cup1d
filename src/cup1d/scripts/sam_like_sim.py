import os, sys, time
import numpy as np
import configargparse

# our own modules
from lace.archive import gadget_archive, nyx_archive
from lace.emulator.nn_emulator import NNEmulator
from lace.emulator.gp_emulator import GPEmulator
from cup1d.data import data_gadget, data_nyx
from cup1d.likelihood import lya_theory, likelihood, emcee_sampler


def parse_args():
    def str_to_bool(s):
        if s == "True":
            return True
        elif s == "False":
            return False

    parser = configargparse.ArgumentParser(
        description="Passing options to sampler"
    )

    # archive and emulator
    parser.add_argument(
        "--training_set",
        default=None,
        choices=["Pedersen21", "Cabayol23", "Nyx23_Oct2023"],
        required=True,
    )

    # emulator
    parser.add_argument(
        "--emulator_label",
        default=None,
        choices=[
            "Pedersen21",
            "Cabayol23",
            "Cabayol23_extended",
            "Nyx_v0",
            "Nyx_v0_extended",
        ],
        required=True,
    )
    parser.add_argument(
        "--test_sim_label",
        default=None,
        type=str,
        required=True,
    )
    parser.add_argument(
        "--drop_sim",
        default="False",
        choices=["True", "False"],
        help="Drop test_sim_label simulation from the training set",
    )

    # P1D
    parser.add_argument(
        "--add_hires",
        default="False",
        choices=["True", "False"],
        help="Include high-res data (Karacayli2022)",
    )
    parser.add_argument(
        "--use_polyfit",
        default="True",
        choices=["True", "False"],
        help="Fit data after fitting polynomial",
    )

    # likelihood
    parser.add_argument(
        "--cov_label",
        type=str,
        default="Chabanier2019",
        help="Data covariance to use, Chabanier2019 or QMLE_Ohio",
    )
    parser.add_argument(
        "--emu_cov_factor",
        type=float,
        default=0,
        help="scale contribution of emulator covariance",
    )
    parser.add_argument(
        "--n_igm",
        type=int,
        default=2,
        help="Number of free parameters for IGM model",
    )
    parser.add_argument(
        "--prior_Gauss_rms",
        default=None,
        help="Width of Gaussian prior",
    )
    # not implemented yet!
    # parser.add_argument(
    #     "--vary_fiducial_cosmo",
    #     default="False",
    #     choices=["True", "False"],
    #     help="Use as fiducial cosmology the one of the target mock",
    # )

    # parser.add_argument(
    #     "--z_min", type=float, default=2.0, help="Minimum redshift"
    # )
    # parser.add_argument(
    #     "--z_max", type=float, default=4.5, help="Maximum redshift"
    # )
    # parser.add_argument(
    #     "--cosmo_fid_label",
    #     type=str,
    #     default="default",
    #     help="Fiducial cosmology to use (default,truth)",
    # )

    #######################
    # print args
    args = parser.parse_args()
    print("--- print options from parser ---")
    print(args)
    # print("----------")
    # print(parser.format_help())
    print("----------")
    print(parser.format_values())
    print("----------")

    args.drop_sim = str_to_bool(args.drop_sim)
    args.add_hires = str_to_bool(args.add_hires)
    args.use_polyfit = str_to_bool(args.use_polyfit)
    args.archive = None

    assert "CUP1D_PATH" in os.environ, "Define CUP1D_PATH variable"

    # # set output dir
    # if args.rootdir:
    #     rootdir = args.rootdir
    #     print("set input rootdir", rootdir)
    # else:
    #
    #     rootdir = os.environ["CUP1D_PATH"] + "/chains/"
    #     print("use default rootdir", rootdir)
    return args


def load_emu(
    archive,
    label_training_set,
    emulator_label,
    test_sim_label,
    drop_sim,
):
    def set_emu_path(
        label_training_set,
        emulator_label,
        test_sim_label,
        drop_sim,
    ):
        folder = "NNmodels/" + label_training_set + "/"
        # set file name
        fname = emulator_label
        if drop_sim != False:
            fname += "_drop_sim_" + test_sim_label
        fname += ".pt"

        return folder + fname

    if emulator_label == "Pedersen21":
        if label_training_set != "Pedersen21":
            raise (
                "Combination of training_set ("
                + label_training_set
                + ") and emulator_label ("
                + emulator_label
                + ") not allowed:"
            )
        if drop_sim:
            _drop_sim = test_sim_label
        else:
            _drop_sim = None
        print("Training emulator " + emulator_label)
        emulator = GPEmulator(
            training_set=label_training_set,
            emulator_label=emulator_label,
            drop_sim=_drop_sim,
        )
    else:
        if (label_training_set == "Cabayol23") & (
            (emulator_label == "Cabayol23")
            | (emulator_label == "Cabayol23_extended")
        ):
            pass
        elif (label_training_set[:5] == "Nyx23") & (
            (emulator_label == "Nyx_v0") | (emulator_label == "Nyx_v0_extended")
        ):
            pass
        else:
            print(
                "Combination of training_set ("
                + label_training_set
                + ") and emulator_label ("
                + emulator_label
                + ") not allowed:"
            )
            sys.exit()

        emu_path = set_emu_path(
            label_training_set, emulator_label, test_sim_label, drop_sim
        )
        if drop_sim:
            _drop_sim = test_sim_label
        else:
            _drop_sim = None
        print("Loading emulator " + emulator_label)
        emulator = NNEmulator(
            archive=archive,
            training_set=label_training_set,
            emulator_label=emulator_label,
            model_path=emu_path,
            drop_sim=_drop_sim,
            train=False,
        )

    return emulator


def path_sampler(args):
    if args.drop_sim:
        flag_drop = "ydrop"
    else:
        flag_drop = "ndrop"
    if args.use_polyfit:
        flag_poly = "ypoly"
    else:
        flag_poly = "npoly"
    if args.add_hires:
        flag_hires = "hres"
    else:
        flag_hires = "lres"

    path = os.path.dirname(os.environ["CUP1D_PATH"]) + "/data/"
    if os.path.isdir(path) == False:
        os.mkdir(path)
    path += "cup1d/"
    if os.path.isdir(path) == False:
        os.mkdir(path)
    path += "sampler/"
    if os.path.isdir(path) == False:
        os.mkdir(path)
    path += args.training_set + "_" + flag_hires + "/"
    if os.path.isdir(path) == False:
        os.mkdir(path)
    path += (
        args.emulator_label
        + "_"
        + args.cov_label
        + "_"
        + args.test_sim_label
        + "_igm"
        + str(args.n_igm)
        + "_"
        + flag_drop
        + "_"
        + flag_poly
        + "/"
    )
    if os.path.isdir(path) == False:
        os.mkdir(path)

    return path


def log_prob(theta):
    return log_prob.sampler.like.log_prob_and_blobs(theta)


def set_log_prob(sampler):
    log_prob.sampler = sampler
    return log_prob


def sample(args, like, free_parameters):
    # cannot call self.log_prob using multiprocess.pool

    path = path_sampler(args)

    sampler = emcee_sampler.EmceeSampler(like=like, rootdir=path)
    _log_prob = set_log_prob(sampler)

    _ = sampler.run_sampler(
        args.n_burn_in, args.n_steps, log_func=_log_prob, parallel=True
    )
    sampler.write_chain_to_file(plots=False)

    listpar = [
        "$\\Delta^2_\\star$",
        "$n_\\star$",
        "$\mathrm{ln}\,\\tau_0$",
        "$\mathrm{ln}\,\\tau_1$",
        "$\mathrm{ln}\,\\sigma^T_0$",
        "$\mathrm{ln}\,\\sigma^T_1$",
        "$\mathrm{ln}\,\\gamma_0$",
        "$\mathrm{ln}\,\\gamma_1$",
        "$\mathrm{ln}\,k^F_0$",
        "$\mathrm{ln}\,k^F_1$",
    ]

    sampler.plot_corner(
        plot_params=["$\\Delta^2_\\star$", "$n_\\star$"],
        # plot_params=listpar,
        delta_lnprob_cut=50,
        usetex=False,
        serif=False,
    )

    # cube_values = np.array(minimizer.minimizer.values)
    # best_fit_values = np.zeros_like(cube_values)
    # err_best_fit_values = np.zeros_like(cube_values)
    # truth_values = np.zeros_like(cube_values) + 0.5
    # for ii, par in enumerate(free_parameters):
    #     # truth
    #     if par in like.truth:
    #         true = like.truth[par]
    #     else:
    #         true = 0.5
    #     truth_values[ii] = true
    #     # best
    #     val_best, err_best = minimizer.best_fit_value(par, return_hesse=True)
    #     # if par == "As":
    #     #     val_best *= 1e-9
    #     #     err_best *= 1e-9
    #     best_fit_values[ii] = val_best
    #     err_best_fit_values[ii] = err_best

    # best_chi2 = like.get_chi2(values=cube_values)
    # print("chi2 improved from {} to {}".format(ini_chi2, best_chi2))
    # # print(best_chi2)
    # # print(free_parameters)
    # # print(truth_values)
    # # print(best_fit_values)
    # # print(err_best_fit_values)

    # if args.archive is None:
    #     out_args = args
    # else:
    #     out_args = args.save()

    # save = {
    #     "metadata": out_args,
    #     "best_chi2": best_chi2,
    #     "name_parameters": free_parameters,
    #     "truth_parameters": truth_values,
    #     "best_parameters": best_fit_values,
    #     "err_best_parameters": err_best_fit_values,
    #     "covariance": np.array(minimizer.minimizer.covariance),
    # }

    # fname = path + "/summary.npy"

    # print("Saving output in:", fname)
    # np.save(fname, save)


def sam_like_sim(args):
    start_all = time.time()

    # os.environ["OMP_NUM_THREADS"] = "1"

    #######################
    # load training set
    start = time.time()
    print("----------")
    print("Setting training set " + args.training_set)

    args.n_steps = 500
    if args.n_igm == 0:
        args.n_burn_in = 75
    else:
        args.n_burn_in = 250

    if args.archive is None:
        if args.training_set == "Pedersen21":
            archive = gadget_archive.GadgetArchive(postproc=args.training_set)
            set_P1D = data_gadget.Gadget_P1D
            z_min = 2
            z_max = 4.5
            # z_max = np.max(archive.list_sim_redshifts)
            sim_igm = "mpg"
            emu_type = "gp"
        elif args.training_set == "Cabayol23":
            archive = gadget_archive.GadgetArchive(postproc=args.training_set)
            set_P1D = data_gadget.Gadget_P1D
            z_min = 2
            z_max = 4.5
            # z_max = np.max(archive.list_sim_redshifts)
            sim_igm = "mpg"
            emu_type = "nn"
        elif args.training_set[:5] == "Nyx23":
            archive = nyx_archive.NyxArchive(nyx_version=args.training_set[6:])
            set_P1D = data_nyx.Nyx_P1D
            z_min = 2.2
            z_max = 4.5
            # z_max = np.max(archive.list_sim_redshifts)
            sim_igm = "nyx"
            emu_type = "nn"
        else:
            raise ValueError("Training_set not implemented")
    else:
        archive = args.archive
        z_min = args.z_min
        z_max = args.z_max
        sim_igm = args.sim_igm
        emu_type = args.emu_type
        set_P1D = args.set_P1D

    if args.test_sim_label not in archive.list_sim:
        print(args.test_sim_label + " is not in part of " + args.training_set)
        print("List of simulations available: ", archive.list_sim)
        sys.exit()
    end = time.time()
    multi_time = str(np.round(end - start, 2))
    print("z in range ", z_min, ", ", z_max)
    print("Training set loaded " + multi_time + " s")

    #######################
    # set emulator
    print("----------")
    print("Setting emulator")
    start = time.time()
    if args.drop_sim:
        ## only drop sim if it was in the training set
        if args.test_sim_label in archive.list_sim_cube:
            _drop_sim = True
        else:
            _drop_sim = False
    else:
        _drop_sim = False
    emulator = load_emu(
        archive,
        args.training_set,
        args.emulator_label,
        args.test_sim_label,
        _drop_sim,
    )
    multi_time = str(np.round(time.time() - start, 2))
    print("Emulator loaded " + multi_time + " s")

    if args.use_polyfit:
        polyfit_kmax_Mpc = emulator.kmax_Mpc
        polyfit_ndeg = emulator.ndeg
    else:
        polyfit_kmax_Mpc = None
        polyfit_ndeg = None

    #######################
    # set target P1D
    data = set_P1D(
        archive=archive,
        sim_label=args.test_sim_label,
        z_min=z_min,
        z_max=z_max,
        data_cov_label=args.cov_label,
        polyfit_kmax_Mpc=polyfit_kmax_Mpc,
        polyfit_ndeg=polyfit_ndeg,
    )
    if args.add_hires:
        extra_data = set_P1D(
            archive=archive,
            sim_label=args.test_sim_label,
            z_min=z_min,
            z_max=z_max,
            data_cov_label="Karacayli2022",
            polyfit_kmax_Mpc=polyfit_kmax_Mpc,
            polyfit_ndeg=polyfit_ndeg,
        )
    else:
        extra_data = None

    #######################
    # set likelihood
    ## set cosmo free parameters
    print("----------")
    print("Set likelihood")
    free_parameters = ["As", "ns"]
    print("Using {} parameters for IGM model".format(args.n_igm))
    for ii in range(args.n_igm):
        for par in ["tau", "sigT_kms", "gamma", "kF"]:
            free_parameters.append("ln_{}_{}".format(par, ii))
    print("free parameters", free_parameters)
    ## set theory
    theory = lya_theory.Theory(
        zs=data.z,
        emulator=emulator,
        free_param_names=free_parameters,
        sim_igm=sim_igm,
        emu_type=emu_type,
    )
    ## set like
    like = likelihood.Likelihood(
        data=data,
        theory=theory,
        free_param_names=free_parameters,
        prior_Gauss_rms=args.prior_Gauss_rms,
        emu_cov_factor=args.emu_cov_factor,
        extra_p1d_data=extra_data,
    )

    #######################
    # sample likelihood
    print("----------")
    print("Sampler")
    start = time.time()
    sample(args, like, free_parameters)
    multi_time = str(np.round(time.time() - start, 2))
    print("Sample in " + multi_time + " s")
    print("")
    print("")
    multi_time = str(np.round(time.time() - start_all, 2))
    print("Program took " + multi_time + " s")
    print("")
    print("")


if __name__ == "__main__":
    args = parse_args()
    sam_like_sim(args)
