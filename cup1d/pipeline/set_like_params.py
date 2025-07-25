def set_free_like_parameters(args, emulator_label="CH24_mpgcen_gpr"):
    """Set free parameters for likelihood"""

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

    # for ii in range(args.fid_cont["n_HCD_const"]):
    #     free_parameters.append(f"HCD_const_{ii}")

    # # SN
    # for ii in range(args.fid_cont["n_sn"]):
    #     free_parameters.append(f"ln_SN_{ii}")

    # # AGN
    # for ii in range(args.fid_cont["n_agn"]):
    #     free_parameters.append(f"ln_AGN_{ii}")

    # Systematics
    for key in args.syst_params:
        for ii in range(args.fid_syst["n_" + key]):
            if key == "res":
                free_parameters.append(f"R_coeff_{ii}")
            else:
                free_parameters.append(f"{key}_{ii}")

    return free_parameters
