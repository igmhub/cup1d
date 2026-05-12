"""Likelihood-parameter selection helpers."""


def set_free_like_parameters(args, emulator_label="CH24_mpgcen_gpr"):
    """Return the free parameter names implied by the pipeline arguments.

    Parameters
    ----------
    args : cup1d.likelihood.input_pipeline.Args
        Pipeline configuration containing cosmology, IGM, contaminant, and
        systematic parameter choices.
    emulator_label : str, optional
        Emulator label used to decide whether Nyx alpha parameters can vary.

    Returns
    -------
    list[str]
        Names of likelihood parameters that should be varied.
    """

    # cosmology
    if args.fix_cosmo:
        free_parameters = []
    else:
        if args.vary_alphas and (
            ("nyx" in emulator_label) | ("Nyx" in emulator_label)
        ):
            free_parameters = ["As", "ns", "nrun"]
        else:
            free_parameters = ["As", "ns"]

    # IGM
    for key in args.igm_params:
        for ii in range(args.fid_igm["n_" + key]):
            free_parameters.append(f"{key}_{ii}")

    # Contaminants
    for key in args.cont_params:
        for ii in range(args.fid_cont["n_" + key]):
            free_parameters.append(f"{key}_{ii}")

    # Systematics
    for key in args.syst_params:
        for ii in range(args.fid_syst["n_" + key]):
            free_parameters.append(f"{key}_{ii}")

    return free_parameters
