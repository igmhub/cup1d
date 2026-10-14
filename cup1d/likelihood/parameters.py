def set_free_likelihood_parameters(args, emulator_label="lace_mpg"):
    """Build the ordered free-parameter list for a configured likelihood.

    Parameters
    ----------
    args : cup1d.configuration.args.Args
        Configuration defining fixed cosmology and IGM, contaminant, and
        systematic parameter families and node counts.
    emulator_label : str, default='lace_mpg'
        Emulator label. Nyx labels permit ``nrun`` when ``vary_alphas`` is
        enabled; other labels vary only ``As`` and ``ns``.

    Returns
    -------
    list of str
        Cosmology parameters followed by indexed IGM, contaminant, and
        systematic parameter names in configuration order.
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
