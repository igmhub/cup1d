from cup1d.p1ds.simulations import (
    data_gadget,
    data_nyx,
    data_accel2,
    challenge_DESIY1,
)

from cup1d.p1ds import forecast

from cup1d.p1ds.observations import (
    data_Walther2018,
    data_Chabanier2019,
    data_Karacayli2022,
    data_Ravoux2023,
    data_Karacayli2024,
    data_DESIY1,
)

from cup1d.emulator.archive import set_archive


def is_synthetic_data_label(data_label):
    """Return whether a P1D label denotes generated rather than observed data.

    Parameters
    ----------
    data_label : str
        Dataset label accepted by :func:`set_p1d`.

    Returns
    -------
    bool
        True for simulation, forecast, acceleration, Sherwood, and DESI
        challenge labels that require a theory prediction.
    """

    return (
        data_label.startswith(("mpg", "nyx", "forecast"))
        or data_label in {"accel2", "sherwood", "challenge_DESIY1"}
    )


def set_p1d(args, data_label, archive=None, theory=None):
    """Construct the P1D data object selected by a configuration label.

    Parameters
    ----------
    args : object
        Parsed Cup1D configuration.  Its covariance, smoothing, noise,
        redshift, and input-file options are used by the selected reader.
    data_label : str
        Label of the simulation, forecast, challenge, or observational data
        product to load.
    archive : object, optional
        Already loaded simulation archive.  A compatible archive is reused;
        otherwise the configured training archive is loaded.
    theory : object, optional
        Theory object required for synthetic labels and used to generate mock
        P1D predictions.

    Returns
    -------
    BaseDataP1D
        Selected measurement or mock data object, with its ``data_label``
        attribute set to ``data_label``.

    Raises
    ------
    ValueError
        If a synthetic label lacks ``theory``, no compatible simulation exists,
        or the label is not implemented.
    """

    if (
        data_label.startswith("mpg")
        | data_label.startswith("nyx")
        | (data_label == "accel2")
        | (data_label == "sherwood")
    ):
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
                archive_mock = set_archive(training_set=args.training_set)
            elif data_label.startswith("nyx"):
                archive_mock = set_archive(training_set=args.training_set)
            elif data_label == "sherwood":
                archive_mock = set_archive(training_set=args.training_set)
            elif data_label == "accel2":
                archive_mock = None
            else:
                raise ValueError("data_label", data_label, "not implemented")

        if data_label != "accel2":
            if data_label not in archive_mock.list_sim:
                raise ValueError(
                    data_label + " not available in archive ",
                    archive_mock.list_sim,
                )
        ##

        # get P1Ds from archive
        if data_label != "accel2":
            p1d_ideal = archive_mock.get_testing_data(data_label)
            if len(p1d_ideal) == 0:
                raise ValueError("Could not set P1D data for", data_label)
            else:
                archive_mock = None
        else:
            p1d_ideal = None

        ## set P1Ds in kms
        if data_label.startswith("mpg"):
            set_p1d_from_mock = data_gadget.Gadget_P1D
        elif data_label.startswith("nyx") | (data_label == "sherwood"):
            set_p1d_from_mock = data_nyx.Nyx_P1D
        elif data_label == "accel2":
            set_p1d_from_mock = data_accel2.Accel2_P1D
        else:
            raise ValueError("data_label", data_label, "not implemented")

        data = set_p1d_from_mock(
            theory,
            p1d_ideal,
            input_sim=data_label,
            data_cov_label=args.synth_cov_label,
            apply_smoothing=args.apply_smoothing,
            add_noise=args.add_noise,
            seed=args.seed_noise,
            z_min=args.z_min,
            z_max=args.z_max,
            path_data=args.path_data,
            p1d_fname=args.p1d_fname,
        )
    elif data_label.startswith("forecast"):
        prefix, rest = data_label.split("_", 1)
        data = forecast.Forecast_P1D(
            theory,
            data_label=rest,
            add_noise=args.add_noise,
            seed=args.seed_noise,
            z_min=args.z_min,
            z_max=args.z_max,
            path_data=args.p1d_fname,
        )
    elif data_label == "challenge_DESIY1":
        data = challenge_DESIY1.P1D_challenge_DESIY1(
            theory,
            p1d_fname=args.p1d_fname,
            z_min=args.z_min,
            z_max=args.z_max,
        )
    elif data_label == "Chabanier2019":
        data = data_Chabanier2019.P1D_Chabanier2019(z_min=args.z_min, z_max=args.z_max)
    elif data_label == "Ravoux2023":
        data = data_Ravoux2023.P1D_Ravoux2023(z_min=args.z_min, z_max=args.z_max)
    elif data_label == "Karacayli2024":
        data = data_Karacayli2024.P1D_Karacayli2024(z_min=args.z_min, z_max=args.z_max)
    elif data_label == "Karacayli2022":
        # The HIRES reader applies its scale selection before constructing the
        # per-redshift and full covariance arrays, keeping them consistent.
        kmax_ikms = getattr(args, "kmax_ikms", None)
        data = data_Karacayli2022.P1D_Karacayli2022(
            kmax_kms=0.1 if kmax_ikms is None else kmax_ikms,
            z_min=args.z_min,
            z_max=args.z_max,
        )
    elif data_label == "Walther2018":
        data = data_Walther2018.P1D_Walther2018(z_min=args.z_min, z_max=args.z_max)
    elif data_label.startswith("DESIY1"):
        data = data_DESIY1.P1D_DESIY1(
            data_label=data_label,
            z_min=args.z_min,
            z_max=args.z_max,
            cov_syst_type=args.cov_syst_type,
            p1d_fname=args.p1d_fname,
            variation=args.name_variation,
            data_bias=args.data_bias,
        )
    else:
        raise ValueError(f"data_label {data_label} not implemented")

    data.data_label = data_label

    return data
