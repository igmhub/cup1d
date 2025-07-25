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
    data_DESIY1,
    challenge_DESIY1,
)

from cup1d.pipeline.set_archive import set_archive
from cup1d.pipeline.set_theory import set_theory


def set_P1D(args, archive=None, theory=None):
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
    """

    data_label = args.data_label

    if data_label.startswith("mpg") | data_label.startswith("nyx"):
        if theory is None:
            raise ValueError("Must provide theory to set P1D from simulation")

        # set P1D from simulation

        ## check if we need to load another archive
        load_archive = False
        if archive is not None:
            if data_label in archive.list_sim:
                archive_mock = archive
            else:
                load_archive = True
        else:
            load_archive = True

        if load_archive:
            if data_label.startswith("mpg"):
                archive_mock = set_archive(training_set="Cabayol23")
            elif data_label.startswith("nyx"):
                archive_mock = set_archive(training_set=args.nyx_training_set)

        if data_label not in archive_mock.list_sim:
            raise ValueError(
                data_label + " not available in archive ",
                archive_mock.list_sim,
            )
        ##

        # get P1Ds from archive
        p1d_ideal = archive_mock.get_testing_data(data_label)
        if len(p1d_ideal) == 0:
            raise ValueError("Could not set P1D data for", data_label)
        else:
            archive_mock = None

        ## set P1Ds in kms
        if data_label.startswith("mpg"):
            set_p1d_from_mock = data_gadget.Gadget_P1D
        elif data_label.startswith("nyx"):
            set_p1d_from_mock = data_nyx.Nyx_P1D

        data = set_p1d_from_mock(
            theory,
            true_cosmo,
            p1d_ideal,
            input_sim=data_label,
            data_cov_label=args.cov_label,
            cov_fname=args.p1d_fname,
            apply_smoothing=args.apply_smoothing,
            add_noise=args.add_noise,
            seed=args.seed_noise,
            z_min=args.z_min,
            z_max=args.z_max,
        )
    elif data_label.startswith("mock"):
        data = mock_data.Mock_P1D(
            theory,
            true_cosmo,
            data_label=data_label[5:],
            add_noise=args.add_noise,
            seed=args.seed_noise,
            z_min=args.z_min,
            z_max=args.z_max,
            p1d_fname=args.p1d_fname,
            cov_only_diag=args.cov_syst_type,
        )

    elif data_label == "challenge_DESIY1":
        data = challenge_DESIY1.P1D_challenge_DESIY1(
            theory,
            true_cosmo,
            p1d_fname=args.p1d_fname,
            z_min=args.z_min,
            z_max=args.z_max,
        )

    elif data_label == "Chabanier2019":
        data = data_Chabanier2019.P1D_Chabanier2019(
            z_min=args.z_min, z_max=args.z_max
        )
    elif data_label == "Ravoux2023":
        data = data_Ravoux2023.P1D_Ravoux2023(
            z_min=args.z_min, z_max=args.z_max
        )
    elif data_label == "Karacayli2024":
        data = data_Karacayli2024.P1D_Karacayli2024(
            z_min=args.z_min, z_max=args.z_max
        )
    elif data_label == "Karacayli2022":
        data = data_Karacayli2022.P1D_Karacayli2022(
            z_min=args.z_min, z_max=args.z_max
        )
    elif data_label == "challenge_v0":
        file = (
            os.environ["CHALLENGE_PATH"]
            + "fiducial_lym1d_p1d_qmleformat_IC.txt"
        )
        data = data_QMLE_Ohio.P1D_QMLE_Ohio(
            filename=file, z_min=args.z_min, z_max=args.z_max
        )
    elif data_label.startswith("DESIY1"):
        data = data_DESIY1.P1D_DESIY1(
            data_label=args.data_label,
            z_min=args.z_min,
            z_max=args.z_max,
            cov_syst_type=args.cov_syst_type,
        )
    else:
        raise ValueError(f"data_label {data_label} not implemented")

    # # cull data within emulator range
    # if cull_data:
    #     if args.true_cosmo_label is not None:
    #         cosmo = set_cosmo(cosmo_label=args.true_cosmo_label)
    #     else:
    #         cosmo = set_cosmo(cosmo_label=args.fid_cosmo_label)

    #     dkms_dMpc_zmin = camb_cosmo.dkms_dMpc(cosmo, z=np.min(data.z))
    #     kmin_kms = emulator.kmin_Mpc / dkms_dMpc_zmin
    #     dkms_dMpc_zmax = camb_cosmo.dkms_dMpc(cosmo, z=np.max(data.z))
    #     kmax_kms = emulator.kmax_Mpc / dkms_dMpc_zmax
    #     data.cull_data(kmin_kms=kmin_kms, kmax_kms=kmax_kms)

    data.data_label = data_label

    return data
