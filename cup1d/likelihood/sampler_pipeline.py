import configargparse
from cup1d.utils.utils import create_print_function, mpi_hello_world

import os, sys, time
import numpy as np
from mpi4py import MPI

# our own modules
import lace
from lace.archive import gadget_archive, nyx_archive
from lace.cosmo import camb_cosmo
from lace.emulator.emulator_manager import set_emulator
from cup1d.p1ds import (
    data_gadget,
    data_nyx,
    data_eBOSS_mock,
    data_Chabanier2019,
    data_Karacayli2022,
    data_Karacayli2023,
    data_Ravoux2023,
)
from cup1d.likelihood import lya_theory, likelihood, emcee_sampler


def parse_args():
    parser = configargparse.ArgumentParser(
        description="Passing options to sampler"
    )

    # emulator
    parser.add_argument(
        "--emulator_label",
        default=None,
        choices=[
            "Pedersen21_ext",
            "Pedersen23_ext",
            "k_bin_sm",
            "Cabayol23",
            "Cabayol23_extended",
            "Nyx_v0",
            "Nyx_v0_extended",
        ],
        required=True,
        help="Type of emulator to be used",
    )
    parser.add_argument(
        "--mock_label",
        default=None,
        type=str,
        required=True,
        help="Input simulation to create mock P1Ds",
    )
    parser.add_argument(
        "--z_min",
        type=float,
        default=2,
        help="Minimum redshift of P1D measurements to be analyzed",
    )
    parser.add_argument(
        "--z_max",
        type=float,
        default=4.5,
        help="Maximum redshift of P1D measurements to be analyzed",
    )
    parser.add_argument(
        "--igm_sim_label",
        default=None,
        type=str,
        required=True,
        help="Input simulation to set fiducial IGM model",
    )
    parser.add_argument(
        "--n_igm",
        type=int,
        default=2,
        help="Number of free parameters for IGM model",
    )
    parser.add_argument(
        "--cosmo_sim_label",
        default=None,
        type=str,
        required=True,
        help="Input simulation to set fiducial cosmology",
    )

    parser.add_argument(
        "--drop_sim",
        action="store_true",
        help="Drop mock_label simulation from the training set",
    )

    # P1D
    parser.add_argument(
        "--add_hires",
        action="store_true",
        help="Include high-res data (Karacayli2022)",
    )
    parser.add_argument(
        "--use_polyfit",
        action="store_true",
        help="Fit data after fitting polynomial",
    )

    # likelihood
    parser.add_argument(
        "--cov_label",
        type=str,
        default="Chabanier2019",
        choices=["Chabanier2019", "QMLE_Ohio"],
        help="Data covariance",
    )
    parser.add_argument(
        "--cov_label_hires",
        type=str,
        default="Karacayli2022",
        choices=["Karacayli2022"],
        help="Data covariance for high-res data",
    )
    parser.add_argument(
        "--add_noise",
        action="store_true",
        help="Add noise to P1D mock according to covariance matrix",
    )
    parser.add_argument(
        "--seed_noise",
        type=int,
        default=0,
        help="Seed for noise",
    )
    parser.add_argument(
        "--fix_cosmo",
        action="store_true",
        help="Fix cosmological parameters while sampling",
    )

    parser.add_argument(
        "--version",
        default="v3",
        help="Version of the pipeline",
    )

    parser.add_argument(
        "--n_steps",
        type=int,
        default=1000,
        help="Steps of emcee chains",
    )
    parser.add_argument(
        "--prior_Gauss_rms",
        default=None,
        help="Width of Gaussian prior",
    )
    parser.add_argument(
        "--emu_cov_factor",
        type=float,
        default=0,
        help="scale contribution of emulator covariance",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print information",
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Run test job",
    )

    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Parallelize",
    )

    #######################
    # print args
    args = parser.parse_args()
    mpi_hello_world()

    fprint = create_print_function(verbose=args.verbose)
    fprint("--- print options from parser ---")
    fprint(args)
    fprint("----------")
    fprint(parser.format_values())
    fprint("----------")

    args.archive = None
    dict_training_set = {
        "Pedersen21": "Pedersen21",
        "Pedersen23": "Cabayol23",
        "Pedersen21_ext": "Cabayol23",
        "Pedersen23_ext": "Cabayol23",
        "k_bin_sm": "Cabayol23",
        "Cabayol23": "Cabayol23",
        "Cabayol23_extended": "Cabayol23",
        "Nyx_v0": "Nyx23_Oct2023",
        "Nyx_v0_extended": "Nyx23_Oct2023",
    }
    args.training_set = training_set_for_emulator[args.emulator_label]

    return args


class SamplerPipeline(object):
    """Full pipeline for extracting cosmology from P1D using sampler"""

    def __init__(self, args):
        ## MPI stuff
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        # create print function (only for rank 0)
        fprint = create_print_function(verbose=args.verbose)
        ###################

        ## set training set (only for rank 0)
        if rank == 0:
            start_all = time.time()
            start = time.time()
            fprint("----------")
            fprint("Setting training set " + args.training_set)
            if args.archive is None:
                # when calling the script from the command line
                archive = self.set_archive(args.training_set)
            else:
                archive = args.archive
            end = time.time()
            multi_time = str(np.round(end - start, 2))
            fprint("Training set loaded in " + multi_time + " s")
        else:
            if args.archive is not None:
                print("WARNING: archive should be None for rank != 0")
                archive = None
        #######################

        ## set emulator
        _drop_sim = None
        if args.drop_sim & (args.mock_label in archive.list_sim_cube):
            _drop_sim = args.mock_label

        if rank == 0:
            fprint("----------")
            fprint("Setting emulator")
            start = time.time()

            emulator = set_emulator(
                emulator_label=args.emulator_label,
                archive=archive,
                drop_sim=_drop_sim,
            )

            multi_time = str(np.round(time.time() - start, 2))
            fprint("Emulator set in " + multi_time + " s")

            # distribute emulator to all ranks
            for irank in range(1, size):
                comm.send(emulator, dest=irank, tag=irank)
        else:
            # receive emulator from ranks 0
            emulator = comm.recv(source=0, tag=rank)

        #######################

        # set P1D
        if rank == 0:
            fprint("----------")
            fprint("Setting P1D")
            start = time.time()

            data = {"P1Ds": None, "extra_P1Ds": None}
            data["P1Ds"] = self.set_P1D(
                archive,
                emulator,
                args.mock_label,
                cov_label=args.cov_label,
                apply_smoothing=args.apply_smoothing,
                z_min=args.z_min,
                z_max=args.z_max,
                add_noise=args.add_noise,
                seed_noise=args.seed_noise,
            )
            if args.add_high_res:
                data["extra_P1Ds"] = self.set_P1D_hires(
                    archive,
                    emulator,
                    args.mock_label_hires,
                    cov_label=args.cov_label_hires,
                    apply_smoothing=args.apply_smoothing,
                    z_min=args.z_min,
                    z_max=args.z_max,
                    add_noise=args.add_noise,
                    seed_noise=args.seed_noise,
                )
            # distribute data to all tasks
            for irank in range(1, size):
                comm.send(data, dest=irank, tag=irank + 1)
        else:
            # get testing_data from task 0
            data = comm.recv(source=0, tag=rank + 1)

        if rank == 0:
            multi_time = str(np.round(time.time() - start, 2))
            fprint("P1D set in " + multi_time + " s")

        # reset archives to free space
        archive = None

        #######################

        # set fiducial cosmology
        cosmo_fid = self.set_fid_cosmo(cosmo_sim_label=args.cosmo_sim_label)

        #######################

        # set likelihood
        fprint("----------")
        fprint("Setting likelihood")

        like = self.set_like(
            emulator,
            data["P1Ds"],
            data["extra_P1Ds"],
            args.mock_label,
            args.igm_sim_label,
            args.n_igm,
            cosmo_fid,
            fix_cosmo=args.fix_cosmo,
            fprint=fprint,
            prior_Gauss_rms=args.prior_Gauss_rms,
            emu_cov_factor=args.emu_cov_factor,
        )

        self.set_emcee_options(
            args.cov_label,
            args.n_igm,
            n_steps=args.n_steps,
            n_burn_in=args.n_burn_in,
            test=args.test,
        )

        #######################

        # set sampler
        if rank == 0:
            fprint("Setting sampler")
            fprint("-------")

            self.out_folder = self.path_sampler(
                args.emulator_label,
                args.mock_label,
                args.igm_sim_label,
                args.n_igm,
                args.cosmo_sim_label,
                args.cov_label,
                version=args.version,
                drop_sim=_drop_sim,
                apply_smoothing=args.apply_smoothing,
                add_hires=args.add_hires,
                add_noise=args.add_noise,
                seed_noise=args.seed_noise,
                fix_cosmo=args.fix_cosmo,
            )

        self.set_sampler(like, fix_cosmo=args.fix_cosmo, parallel=args.parallel)

        #######################

        if rank == 0:
            multi_time = str(np.round(time.time() - start_all, 2))
            fprint("Setting the sampler took " + multi_time + " s \n\n")

    def path_sampler(
        self,
        emulator_label,
        mock_label,
        igm_sim_label,
        n_igm,
        cosmo_sim_label,
        cov_label,
        version="v3",
        drop_sim=None,
        apply_smoothing=True,
        add_hires=False,
        add_noise=False,
        seed_noise=0,
        fix_cosmo=False,
    ):
        if drop_sim is not None:
            flag_drop = "_drop"
        else:
            flag_drop = ""

        if apply_smoothing:
            flag_smooth = "_smooth"
        else:
            flag_smooth = ""

        if add_hires:
            flag_hires = "_hires"
        else:
            flag_hires = ""

        try:
            path = os.environ["LYA_DATA_PATH"]
        except:
            raise ValueError("LYA_DATA_PATH not set as environment variable")

        if os.path.isdir(path) == False:
            os.mkdir(path)
        path += "cup1d/"
        if os.path.isdir(path) == False:
            os.mkdir(path)
        path += "sampler/"
        if os.path.isdir(path) == False:
            os.mkdir(path)
        path += version + "/"
        if os.path.isdir(path) == False:
            os.mkdir(path)
        path += "emu_" + emulator_label + "/"
        if os.path.isdir(path) == False:
            os.mkdir(path)
        path += "cov_" + cov_label + flag_hires + "/"
        if os.path.isdir(path) == False:
            os.mkdir(path)

        path += (
            "mock_"
            + mock_label
            + "_igm_"
            + igm_sim_label
            + "_cosmo_"
            + cosmo_sim_label
            + "_nigm_"
            + str(n_igm)
            + flag_drop
            + flag_smooth
        )

        if add_noise:
            path += "_noise_" + str(seed_noise)
        if fix_cosmo:
            path += "_fix_cosmo"
        path += "/"

        if os.path.isdir(path) == False:
            os.mkdir(path)

        return path

    def set_archive(self, training_set):
        if (training_set == "Pedersen21") | (training_set == "Cabayol23"):
            archive = gadget_archive.GadgetArchive(postproc=training_set)
        elif training_set[:3] == "Nyx":
            archive = nyx_archive.NyxArchive(nyx_version=training_set[6:])
        else:
            raise ValueError(
                "Training set " + training_set + " not implemented"
            )
        return archive

    def set_emcee_options(
        self, cov_label, n_igm, n_steps=None, n_burn_in=None, test=False
    ):
        if test == True:
            self.n_steps = 10
            self.n_burn_in = 0
        else:
            if n_steps is not None:
                self.n_steps = n_steps
            else:
                self.n_steps = 1000

            if n_burn_in is not None:
                self.n_burn_in = n_burn_in
            else:
                if cov_label == "Chabanier2019":
                    if n_igm == 0:
                        self.n_burn_in = 100
                    else:
                        self.n_burn_in = 500
                elif cov_label == "QMLE_Ohio":
                    if n_igm == 0:
                        self.n_burn_in = 200
                    else:
                        self.n_burn_in = 1200
                else:
                    if n_igm == 0:
                        self.n_burn_in = 100
                    else:
                        self.n_burn_in = 500

    def set_P1D(
        self,
        archive,
        emulator,
        mock_label,
        cov_label=None,
        apply_smoothing=None,
        z_min=0,
        z_max=10,
        add_noise=False,
        seed_noise=0,
    ):
        """Set P1D data

        Parameters
        ----------
        archive : object
            Archive object containing P1D data
        mock_label : str
            Label of simulation/dataset used to generate mock data
        cov_label : str, optional
            Label of covariance matrix
        apply_smoothing : bool or None
            If True, apply smoothing to P1D. If None, do what is best for the input emulator
        z_min : float
            Minimum redshift of P1D measurements
        z_max : float
            Maximum redshift of P1D measurements
        """

        if (mock_label[:3] == "mpg") | (mock_label[:3] == "nyx"):
            # check if we need to load another archive
            if mock_label in archive.list_sim:
                archive_mock = archive
            else:
                if mock_label[:3] == "mpg":
                    archive_mock = set_archive(training_set="Cabayol23")
                elif mock_label[:3] == "nyx":
                    archive_mock = set_archive(training_set="Nyx24_Feb2024")

            if mock_label not in archive_mock.list_sim:
                raise ValueError(
                    mock_label + " not available in archive ",
                    archive_mock.list_sim,
                )
            ###################

            # set noise free P1Ds in Mpc
            p1d_ideal = archive_mock.get_testing_data(mock_label)
            if len(p1d_ideal) == 0:
                raise ValueError("Could not set P1D data for", mock_label)
            else:
                archive_mock = None
            ###################

            # set P1Ds in kms
            if mock_label[:3] == "mpg":
                set_p1d_from_mock = data_gadget.Gadget_P1D
            elif mock_label[:3] == "nyx":
                set_p1d_from_mock = data_nyx.Nyx_P1D

            data = set_p1d_from_mock(
                z_min=z_min,
                z_max=z_max,
                testing_data=p1d_ideal,
                input_sim=mock_label,
                data_cov_label=cov_label,
                emulator=emulator,
                apply_smoothing=apply_smoothing,
                add_noise=add_noise,
                seed=seed_noise,
            )

        elif mock_label == "eBOSS_mock":
            data = data_eBOSS_mock.P1D_eBOSS_mock(
                z_min=z_min,
                z_max=z_max,
                emulator=emulator,
                apply_smoothing=apply_smoothing,
                add_noise=add_noise,
                seed=seed_noise,
            )
        elif mock_label == "Chabanier19":
            data = data_Chabanier2019.P1D_Chabanier2019(
                z_min=z_min,
                z_max=z_max,
                emulator=emulator,
                apply_smoothing=apply_smoothing,
            )
        elif mock_label == "Ravoux23":
            data = data_Ravoux2023.P1D_Ravoux23(
                z_min=z_min,
                z_max=z_max,
                emulator=emulator,
                apply_smoothing=apply_smoothing,
            )
        elif mock_label == "Karacayli23":
            data = data_Karacayli2023.P1D_Karacayli2023(
                z_min=z_min,
                z_max=z_max,
                emulator=emulator,
                apply_smoothing=apply_smoothing,
            )
        else:
            raise ValueError(f"mock_label {mock_label} not implemented")

        return data

    def set_P1D_hires(
        self,
        archive,
        emulator,
        mock_label_hires,
        extra_cov_label,
        apply_smoothing=None,
        z_min=0,
        z_max=10,
        add_noise=False,
        seed_noise=0,
    ):
        """Set P1D data

        Parameters
        ----------
        archive : object
            Archive object containing P1D data
        mock_label : str
            Label of simulation/dataset used to generate mock data
        cov_label : str
            Label of covariance matrix
        apply_smoothing : bool or None
            If True, apply smoothing to P1D. If None, do what is best for the input emulator
        z_min : float
            Minimum redshift of P1D measurements
        z_max : float
            Maximum redshift of P1D measurements
        """

        if (mock_label[:3] == "mpg") | (mock_label[:3] == "nyx"):
            # check if we need to load another archive
            if mock_label in archive.list_sim:
                archive_mock = archive
            else:
                if mock_label[:3] == "mpg":
                    archive_mock = set_archive(training_set="Cabayol23")
                elif mock_label[:3] == "nyx":
                    archive_mock = set_archive(training_set="Nyx24_Feb2024")

            if mock_label not in archive_mock.list_sim:
                raise ValueError(
                    mock_label + " not available in archive ",
                    archive_mock.list_sim,
                )
            ###################

            # set noise free P1Ds in Mpc
            p1d_ideal = archive_mock.get_testing_data(
                mock_label, z_min=z_min, z_max=z_max
            )
            if len(p1d_ideal) == 0:
                raise ValueError("Could not set P1D data for", mock_label)
            else:
                archive_mock = None
            ###################

            # set P1Ds in kms
            if mock_label[:3] == "mpg":
                set_p1d_from_mock = data_gadget.Gadget_P1D
            elif mock_label[:3] == "nyx":
                set_p1d_from_mock = data_nyx.Nyx_P1D

            data_hires = set_p1d_from_mock(
                z_min=z_min,
                z_max=z_max,
                testing_data=p1d_ideal,
                input_sim=mock_label_hires,
                data_cov_label=extra_cov_label,
                emulator=emulator,
                apply_smoothing=apply_smoothing,
                add_noise=add_noise,
                seed=seed_noise,
            )

        elif mock_label_hires == "Karacayli22":
            data_hires = data_Karacayli2022.P1D_Karacayli2022(
                z_min=z_min,
                z_max=z_max,
                emulator=emulator,
                apply_smoothing=apply_smoothing,
            )
        else:
            raise ValueError(
                f"mock_label_hires {mock_label_hires} not implemented"
            )

        return data_hires

    def set_fid_cosmo(self, cosmo_sim_label="mpg_central"):
        if (cosmo_sim_label[:3] == "mpg") | (cosmo_sim_label[:3] == "nyx"):
            if cosmo_sim_label[:3] == "mpg":
                repo = os.path.dirname(lace.__path__[0]) + "/"
                fname = repo + ("data/sim_suites/Australia20/mpg_emu_cosmo.npy")
                get_cosmo = camb_cosmo.get_cosmology_from_dictionary
            elif cosmo_sim_label[:3] == "nyx":
                fname = os.environ["NYX_PATH"] + "nyx_emu_cosmo_Oct2023.npy"
                get_cosmo = camb_cosmo.get_Nyx_cosmology

            try:
                data_cosmo = np.load(fname, allow_pickle=True)
            except:
                ValueError(f"{fname} not found")

            cosmo_fid = None
            for ii in range(len(data_cosmo)):
                if data_cosmo[ii]["sim_label"] == cosmo_sim_label:
                    cosmo_fid = get_cosmo(data_cosmo[ii]["cosmo_params"])
                    break
            if cosmo_fid is None:
                raise ValueError(
                    f"Cosmo not found in {fname} for {cosmo_sim_label}"
                )
        else:
            raise ValueError(
                f"cosmo_sim_label {cosmo_sim_label} not implemented"
            )
        return cosmo_fid

    def set_like(
        self,
        emulator,
        data,
        data_hires,
        mock_label,
        igm_sim_label,
        n_igm,
        cosmo_fid,
        fix_cosmo=False,
        fprint=print,
        prior_Gauss_rms=None,
        emu_cov_factor=0,
    ):
        ## set cosmo and IGM parameters
        if fix_cosmo:
            free_parameters = []
        else:
            free_parameters = ["As", "ns"]

        fprint(f"Using {n_igm} parameters for IGM model")
        for ii in range(n_igm):
            for par in ["tau", "sigT_kms", "gamma", "kF"]:
                free_parameters.append(f"ln_{par}_{ii}")

        fprint("free parameters", free_parameters)

        ## set theory
        theory = lya_theory.Theory(
            zs=data.z,
            emulator=emulator,
            free_param_names=free_parameters,
            fid_sim_igm=igm_sim_label,
            true_sim_igm=mock_label,
            cosmo_fid=cosmo_fid,
        )

        ## set like
        like = likelihood.Likelihood(
            data=data,
            theory=theory,
            free_param_names=free_parameters,
            prior_Gauss_rms=prior_Gauss_rms,
            emu_cov_factor=emu_cov_factor,
            extra_p1d_data=data_hires,
        )

        return like

    def set_sampler(self, like, fix_cosmo=False, parallel=True):
        """Sample the posterior distribution"""

        def log_prob(theta):
            return log_prob.sampler.like.log_prob_and_blobs(theta)

        def set_log_prob(sampler):
            log_prob.sampler = sampler
            return log_prob

        self.sampler = emcee_sampler.EmceeSampler(
            like=like,
            rootdir=self.out_folder,
            save_chain=False,
            nburnin=self.n_burn_in,
            nsteps=self.n_steps,
            parallel=parallel,
            fix_cosmology=fix_cosmo,
        )
        self._log_prob = set_log_prob(self.sampler)

    def run_sampler(self):
        self.sampler.run_sampler(log_func=self._log_prob)

        if MPI.COMM_WORLD.Get_rank() == 0:
            self.sampler.write_chain_to_file()
