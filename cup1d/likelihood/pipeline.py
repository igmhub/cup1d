from cup1d.utils.utils import create_print_function

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
    data_Karacayli2024,
    data_Ravoux2023,
    data_QMLE_Ohio,
    mock_data,
)
from cup1d.likelihood import lya_theory, likelihood, fitter


def set_free_like_parameters(params):
    if params.fix_cosmo:
        free_parameters = []
    else:
        if params.vary_alphas:
            free_parameters = ["As", "ns", "nrun"]
        else:
            free_parameters = ["As", "ns"]

    for ii in range(params.n_igm):
        for par in ["tau", "sigT_kms", "gamma", "kF"]:
            free_parameters.append(f"ln_{par}_{ii}")
    for ii in range(params.n_metals):
        free_parameters.append(f"ln_SiIII_{ii}")
        free_parameters.append(f"ln_SiII_{ii}")
    for ii in range(params.n_dla):
        free_parameters.append(f"ln_A_damp_{ii}")
    for ii in range(params.n_sn):
        free_parameters.append(f"ln_SN_{ii}")

    return free_parameters


def set_archive(training_set):
    """Set archive

    Parameters
    ----------
    training_set : str

    Returns
    -------
    archive : object

    """
    if training_set[:3] == "Nyx":
        archive = nyx_archive.NyxArchive(nyx_version=training_set[6:])
    else:
        archive = gadget_archive.GadgetArchive(postproc=training_set)
    return archive


def set_P1D(
    archive,
    emulator,
    data_label,
    true_cosmo,
    true_sim_igm=None,
    cov_label=None,
    apply_smoothing=False,
    z_min=0,
    z_max=10,
    add_noise=False,
    seed_noise=0,
    fprint=print,
    cull_data=False,
    true_SiII=-10,
    true_SiIII=-10,
    true_HCD=-6,
    true_SN=-10,
):
    """Set P1D data

    Parameters
    ----------
    archive : object
        Archive object containing P1D data
    data_label : str
        Label of simulation/dataset used to generate mock data
    cov_label : str, optional
        Label of covariance matrix
    apply_smoothing : bool or None
        If True, apply smoothing to P1D. If None, do what is best for the input emulator
    z_min : float
        Minimum redshift of P1D measurements
    z_max : float
        Maximum redshift of P1D measurements
    cull_data : bool
        If True, cull data outside of k range from emulator

    Returns
    -------
    data : object
        P1D data
    true_sim_igm : str
        Label of simulation/dataset used to generate mock data
    """

    if (data_label[:3] == "mpg") | (data_label[:3] == "nyx"):
        # check if we need to load another archive
        if data_label in archive.list_sim:
            archive_mock = archive
        else:
            if data_label[:3] == "mpg":
                archive_mock = set_archive(training_set="Cabayol23")
            elif data_label[:3] == "nyx":
                archive_mock = set_archive(training_set="Nyx24_Oct2023")

        if data_label not in archive_mock.list_sim:
            raise ValueError(
                data_label + " not available in archive ",
                archive_mock.list_sim,
            )
        ###################

        # set noise free P1Ds in Mpc
        p1d_ideal = archive_mock.get_testing_data(data_label)
        if len(p1d_ideal) == 0:
            raise ValueError("Could not set P1D data for", data_label)
        else:
            archive_mock = None
        ###################

        # set P1Ds in kms
        if data_label[:3] == "mpg":
            set_p1d_from_mock = data_gadget.Gadget_P1D
        elif data_label[:3] == "nyx":
            set_p1d_from_mock = data_nyx.Nyx_P1D

        data = set_p1d_from_mock(
            z_min=z_min,
            z_max=z_max,
            testing_data=p1d_ideal,
            input_sim=data_label,
            data_cov_label=cov_label,
            emulator=emulator,
            apply_smoothing=apply_smoothing,
            add_noise=add_noise,
            seed=seed_noise,
            true_SiII=true_SiII,
            true_SiIII=true_SiIII,
            true_HCD=true_HCD,
            true_SN=true_SN,
        )
    elif data_label[:5] == "mock_":
        # mock data from emulator
        data = mock_data.Mock_P1D(
            emulator,
            data_label=data_label[5:],
            true_sim_igm=true_sim_igm,
            true_SiII=true_SiII,
            true_SiIII=true_SiIII,
            true_HCD=true_HCD,
            true_SN=true_SN,
            z_min=z_min,
            z_max=z_max,
            add_noise=add_noise,
            seed=seed_noise,
            true_cosmo=true_cosmo,
        )
    elif data_label == "eBOSS_mock":
        # need to be tested
        data = data_eBOSS_mock.P1D_eBOSS_mock(
            z_min=z_min,
            z_max=z_max,
            emulator=emulator,
            apply_smoothing=apply_smoothing,
            add_noise=add_noise,
            seed=seed_noise,
        )
    elif data_label == "Chabanier2019":
        data = data_Chabanier2019.P1D_Chabanier2019(z_min=z_min, z_max=z_max)
    elif data_label == "Ravoux2023":
        data = data_Ravoux2023.P1D_Ravoux2023(z_min=z_min, z_max=z_max)
    elif data_label == "Karacayli2024":
        data = data_Karacayli2024.P1D_Karacayli2024(z_min=z_min, z_max=z_max)
    elif data_label == "Karacayli2022":
        data = data_Karacayli2022.P1D_Karacayli2022(z_min=z_min, z_max=z_max)
    elif data_label == "challenge_v0":
        file = (
            os.environ["CHALLENGE_PATH"]
            + "fiducial_lym1d_p1d_qmleformat_IC.txt"
        )
        data = data_QMLE_Ohio.P1D_QMLE_Ohio(
            filename=file, z_min=z_min, z_max=z_max
        )
    else:
        raise ValueError(f"data_label {data_label} not implemented")

    if cull_data:
        dkms_dMpc_zmin = camb_cosmo.dkms_dMpc(true_cosmo, z=np.min(data.z))
        kmin_kms = emulator.kmin_Mpc / dkms_dMpc_zmin
        dkms_dMpc_zmax = camb_cosmo.dkms_dMpc(true_cosmo, z=np.max(data.z))
        kmax_kms = emulator.kmax_Mpc / dkms_dMpc_zmax
        data.cull_data(kmin_kms=kmin_kms, kmax_kms=kmax_kms)

    return data


def set_cosmo_from_sim(cosmo_label="mpg_central"):
    """Set fiducial cosmology

    Parameters
    ----------
    cosmo_label : str

    Returns
    -------
    cosmo : object
    """
    if (cosmo_label[:3] == "mpg") | (cosmo_label[:3] == "nyx"):
        if cosmo_label[:3] == "mpg":
            repo = os.path.dirname(lace.__path__[0]) + "/"
            fname = repo + ("data/sim_suites/Australia20/mpg_emu_cosmo.npy")
            get_cosmo = camb_cosmo.get_cosmology_from_dictionary
        elif cosmo_label[:3] == "nyx":
            fname = os.environ["NYX_PATH"] + "nyx_emu_cosmo_Oct2023.npy"
            get_cosmo = camb_cosmo.get_Nyx_cosmology

        try:
            data_cosmo = np.load(fname, allow_pickle=True)
        except:
            ValueError(f"{fname} not found")

        cosmo = None
        for ii in range(len(data_cosmo)):
            if data_cosmo[ii]["sim_label"] == cosmo_label:
                cosmo = get_cosmo(data_cosmo[ii]["cosmo_params"])
                break
        if cosmo is None:
            raise ValueError(f"Cosmo not found in {fname} for {cosmo_label}")
    else:
        raise ValueError(f"cosmo_label {cosmo_label} not implemented")
    return cosmo


def set_like(
    emulator,
    data,
    data_hires,
    fid_igm_label,
    free_parameters,
    fid_cosmo,
    fix_cosmo=False,
    vary_alphas=False,
    fid_SiIII=-10,
    fid_SiII=-10,
    fid_HCD=-6,
    fid_SN=-10,
    fprint=print,
    prior_Gauss_rms=None,
    emu_cov_factor=0,
):
    ## set theory
    if data_hires is not None:
        zs_hires = data_hires.z
    else:
        zs_hires = None
    theory = lya_theory.Theory(
        zs=data.z,
        zs_hires=zs_hires,
        emulator=emulator,
        free_param_names=free_parameters,
        fid_sim_igm=fid_igm_label,
        fid_cosmo=fid_cosmo,
        fid_SiIII=fid_SiIII,
        fid_SiII=fid_SiII,
        fid_HCD=fid_HCD,
        fid_SN=fid_SN,
    )

    ## set like
    like = likelihood.Likelihood(
        data,
        theory,
        extra_data=data_hires,
        free_param_names=free_parameters,
        prior_Gauss_rms=prior_Gauss_rms,
        emu_cov_factor=emu_cov_factor,
    )

    return like


def path_sampler(
    emulator_label,
    data_label,
    igm_label,
    n_igm,
    cosmo_label,
    cov_label,
    version="v3",
    drop_sim=None,
    apply_smoothing=True,
    data_label_hires=False,
    add_noise=False,
    seed_noise=0,
    fix_cosmo=False,
    vary_alphas=False,
):
    if drop_sim is not None:
        flag_drop = "_drop"
    else:
        flag_drop = ""

    if apply_smoothing:
        flag_smooth = "_smooth"
    else:
        flag_smooth = ""

    if data_label_hires:
        flag_hires = "_" + data_label_hires
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
        + data_label
        + "_igm_"
        + igm_label
        + "_cosmo_"
        + cosmo_label
        + "_nigm_"
        + str(n_igm)
        + flag_drop
        + flag_smooth
    )

    if add_noise:
        path += "_noise_" + str(seed_noise)
    if fix_cosmo:
        path += "_fix_cosmo"
    if vary_alphas:
        path += "_vary_alphas"
    path += "/"

    if os.path.isdir(path) == False:
        os.mkdir(path)

    return path


class Pipeline(object):
    """Full pipeline for extracting cosmology from P1D using sampler"""

    def __init__(self, args):
        ## MPI stuff
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        # create print function (only for rank 0)
        fprint = create_print_function(verbose=args.verbose)
        self.fprint = fprint
        self.explore = args.explore
        ###################

        ## set training set (only for rank 0)
        if rank == 0:
            start_all = time.time()
            start = time.time()
            fprint("----------")
            fprint("Setting training set " + args.training_set)
            if args.archive is None:
                # when calling the script from the command line
                archive = set_archive(args.training_set)
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
        if rank == 0:
            fprint("----------")
            fprint("Setting emulator")
            start = time.time()

            _drop_sim = None
            if args.drop_sim & (args.data_label in archive.list_sim_cube):
                _drop_sim = args.data_label

            emulator = set_emulator(
                emulator_label=args.emulator_label,
                archive=archive,
                drop_sim=_drop_sim,
            )

            multi_time = str(np.round(time.time() - start, 2))
            fprint("Emulator set in " + multi_time + " s")

            # distribute emulator to all ranks
            for irank in range(1, size):
                comm.send(emulator, dest=irank, tag=(irank + 1) * 7)
        else:
            # receive emulator from ranks 0
            emulator = comm.recv(source=0, tag=(rank + 1) * 7)

        #######################

        ## set fiducial cosmology
        fid_cosmo = set_fid_cosmo(cosmo_label=args.cosmo_label)

        #######################

        ## set P1D
        if rank == 0:
            fprint("----------")
            fprint("Setting P1D")
            start = time.time()

            data = {"P1Ds": None, "extra_P1Ds": None}
            data["P1Ds"] = set_P1D(
                archive,
                emulator,
                args.data_label,
                fid_cosmo,
                true_sim_igm=args.igm_label,
                cov_label=args.cov_label,
                apply_smoothing=args.apply_smoothing,
                z_min=args.z_min,
                z_max=args.z_max,
                add_noise=args.add_noise,
                seed_noise=args.seed_noise,
                fprint=fprint,
            )
            fprint(
                "Set " + str(len(data["P1Ds"].z)) + " P1Ds at z = ",
                data["P1Ds"].z,
            )
            if args.data_label_hires is not None:
                data["extra_P1Ds"] = set_P1D_hires(
                    archive,
                    emulator,
                    args.data_label_hires,
                    fid_cosmo,
                    true_sim_igm=args.igm_label,
                    cov_label=args.cov_label_hires,
                    apply_smoothing=args.apply_smoothing,
                    z_min=args.z_min,
                    z_max=args.z_max,
                    add_noise=args.add_noise,
                    seed_noise=args.seed_noise,
                    fprint=fprint,
                )
                fprint(
                    "Set " + str(len(data["extra_P1Ds"].z)) + " P1Ds at z = ",
                    data["extra_P1Ds"].z,
                )
            # distribute data to all tasks
            for irank in range(1, size):
                comm.send(data, dest=irank, tag=(irank + 1) * 11)
        else:
            # get testing_data from task 0
            data = comm.recv(source=0, tag=(rank + 1) * 11)

        if rank == 0:
            multi_time = str(np.round(time.time() - start, 2))
            fprint("P1D set in " + multi_time + " s")

        #######################

        # set likelihood
        fprint("----------")
        fprint("Setting likelihood")

        like = set_like(
            emulator,
            data["P1Ds"],
            data["extra_P1Ds"],
            args.igm_label,
            args.n_igm,
            fid_cosmo,
            fix_cosmo=args.fix_cosmo,
            vary_alphas=args.vary_alphas,
            fprint=fprint,
            prior_Gauss_rms=args.prior_Gauss_rms,
            emu_cov_factor=args.emu_cov_factor,
        )

        self.set_emcee_options(
            args.data_label,
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

            self.out_folder = path_sampler(
                args.emulator_label,
                args.data_label,
                args.igm_label,
                args.n_igm,
                args.cosmo_label,
                args.cov_label,
                version=args.version,
                drop_sim=_drop_sim,
                apply_smoothing=args.apply_smoothing,
                data_label_hires=args.data_label_hires,
                add_noise=args.add_noise,
                seed_noise=args.seed_noise,
                fix_cosmo=args.fix_cosmo,
                vary_alphas=args.vary_alphas,
            )

            # distribute out_folder to all tasks
            for irank in range(1, size):
                comm.send(self.out_folder, dest=irank, tag=(irank + 1) * 13)
        else:
            # get testing_data from task 0
            self.out_folder = comm.recv(source=0, tag=(rank + 1) * 13)

        self.set_sampler(like, fix_cosmo=args.fix_cosmo, parallel=args.parallel)

        #######################

        if rank == 0:
            multi_time = str(np.round(time.time() - start_all, 2))
            fprint("Setting the sampler took " + multi_time + " s \n\n")

    def set_emcee_options(
        self,
        data_label,
        cov_label,
        n_igm,
        n_steps=0,
        n_burn_in=0,
        test=False,
    ):
        # set steps
        if test == True:
            self.n_steps = 10
        else:
            if n_steps != 0:
                self.n_steps = n_steps
            else:
                if data_label == "Chabanier2019":
                    self.n_steps = 2000
                else:
                    self.n_steps = 1250

        # set burn-in
        if test == True:
            self.n_burn_in = 0
        else:
            if n_burn_in != 0:
                self.n_burn_in = n_burn_in
            else:
                if data_label == "Chabanier2019":
                    self.n_burn_in = 2000
                else:
                    if cov_label == "Chabanier2019":
                        self.n_burn_in = 1500
                    elif cov_label == "QMLE_Ohio":
                        self.n_burn_in = 1500
                    else:
                        self.n_burn_in = 1500

    def set_sampler(self, like, fix_cosmo=False, parallel=True):
        """Sample the posterior distribution"""

        def log_prob(theta):
            return log_prob.sampler.like.log_prob_and_blobs(theta)

        def set_log_prob(sampler):
            log_prob.sampler = sampler
            return log_prob

        self.fitter = fitter.EmceeSampler(
            like=like,
            rootdir=self.out_folder,
            save_chain=False,
            nburnin=self.n_burn_in,
            nsteps=self.n_steps,
            parallel=parallel,
            explore=self.explore,
            fix_cosmology=fix_cosmo,
        )
        self._log_prob = set_log_prob(self.fitter)

    def run_sampler(self):
        ### XXX put minimizer outsize
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        if rank == 0:
            start = time.time()
            self.fprint("----------")
            self.fprint("Running sampler")

        self.sampler.run_sampler(log_func=self._log_prob)

        if rank == 0:
            end = time.time()
            multi_time = str(np.round(end - start, 2))
            self.fprint("Sampler run in " + multi_time + " s")

            # improve best fit from sampler using minimize

            self.fprint("----------")
            self.fprint("Running minimizer")
            start = time.time()
            ind = np.argmax(self.sampler.lnprob.reshape(-1))
            nparam = self.sampler.chain.shape[-1]
            p0 = self.sampler.chain.reshape(-1, nparam)[ind, :]
            self.sampler.run_minimizer(log_func=self._log_prob, p0=p0)
            end = time.time()
            multi_time = str(np.round(end - start, 2))
            self.fprint("Minimizer run in " + multi_time + " s")

            start = time.time()
            self.fprint("----------")
            self.fprint("Saving data")
            self.sampler.write_chain_to_file()
            end = time.time()
            multi_time = str(np.round(end - start, 2))
            self.fprint("Data saved in " + multi_time + " s")
