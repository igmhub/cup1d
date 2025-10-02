## Dictionary to convert likelihood parameters into latex strings
param_dict = {
    "Delta2_p": r"$\Delta^2_p$",
    "mF": r"$F$",
    "gamma": r"$\gamma$",
    "sigT_Mpc": r"$\sigma_T$",
    "kF_Mpc": r"$k_F$",
    "n_p": r"$n_p$",
    "Delta2_star": r"$\Delta^2_\star$",
    "n_star": r"$n_\star$",
    "alpha_star": r"$\alpha_\star$",
    "g_star": r"$g_\star$",
    "f_star": r"$f_\star$",
    "H0": r"$H_0$",
    "mnu": r"$\Sigma m_\nu$",
    "As": r"$A_s$",
    "ns": r"$n_s$",
    "nrun": r"$n_\mathrm{run}$",
    "ombh2": r"$\omega_b$",
    "omch2": r"$\omega_c$",
    "cosmomc_theta": r"$\theta_{MC}$",
}

hcd_dict = ["LLS", "sub-DLA", "small DLA", "large DLA"]

for ii in range(11):
    param_dict["tau_eff_" + str(ii)] = r"$f_{\tau_{\rm eff_" + str(ii) + "}}$"
    param_dict["sigT_kms_" + str(ii)] = r"$f_{\sigma_{\rm T_" + str(ii) + "}}$"
    param_dict["gamma_" + str(ii)] = r"$f_{\gamma_" + str(ii) + r"}$"
    param_dict["kF_kms_" + str(ii)] = r"$f_{{{k_F}_" + str(ii) + r"}}$"
    param_dict["R_coeff_" + str(ii)] = r"$R_" + str(ii) + r"$"
    param_dict["ln_SN_" + str(ii)] = r"$\log \mathrm{SN}_" + str(ii) + r"$"
    param_dict["ln_AGN_" + str(ii)] = r"$\log \mathrm{AGN}_" + str(ii) + r"$"
    for jj in range(1, 5):
        param_dict["HCD_damp" + str(jj) + "_" + str(ii)] = (
            r"$f^\mathrm{HCD}_{{\rm "
            + hcd_dict[jj - 1]
            + "}_"
            + str(ii)
            + r"}$"
        )
    param_dict["HCD_const_" + str(ii)] = r"$c_{{\rm HCD}_" + str(ii) + r"}$"


metal_lines = [
    "Lya_SiIII",
    "Lya_SiII",
    "SiIIa_SiIIb",
    "SiIIa_SiIII",
    "SiIIb_SiIII",
    "CIVa_CIVb",
    "MgIIa_MgIIb",
]
metal_lines_latex_out = {
    "Lya_SiIII": r"$\mathrm{Ly}\alpha-\mathrm{SiIII}$",
    "Lya_SiII": r"$\mathrm{Ly}\alpha-\mathrm{SiII}$",
    "Lya_SiIIa": r"$\mathrm{Ly}\alpha-\mathrm{SiIIa}$",
    "Lya_SiIIb": r"$\mathrm{Ly}\alpha-\mathrm{SiIIb}$",
    "Lya_SiIIc": r"$\mathrm{Ly}\alpha-\mathrm{SiIIc}$",
    "SiIIa_SiIIb": r"$\mathrm{SiIIa}_\mathrm{SiIIb}$",
    "SiIIa_SiIII": r"$\mathrm{SiIIa}_\mathrm{SiIII}$",
    "SiIIb_SiIII": r"$\mathrm{SiIIb}_\mathrm{SiIII}$",
    "SiII_SiIII": r"$\mathrm{SiII}_\mathrm{SiIII}$",
    "SiIIc_SiIII": r"$\mathrm{SiIIc}_\mathrm{SiIII}$",
    "CIVa_CIVb": r"$\mathrm{CIVa}_\mathrm{CIVb}$",
    "MgIIa_MgIIb": r"$\mathrm{MgIIa}_\mathrm{MgIIb}$",
}
metal_lines_latex = {
    "Lya_SiIII": r"\mathrm{Ly}\alpha-\mathrm{SiIII}",
    "Lya_SiII": r"\mathrm{Ly}\alpha-\mathrm{SiII}",
    "Lya_SiIIa": r"\mathrm{Ly}\alpha-\mathrm{SiIIa}",
    "Lya_SiIIb": r"\mathrm{Ly}\alpha-\mathrm{SiIIb}",
    "Lya_SiIIc": r"\mathrm{Ly}\alpha-\mathrm{SiIIc}",
    "SiIIa_SiIIb": r"\mathrm{SiII}-\mathrm{SiII}",
    # "SiIIa_SiIII": r"\mathrm{SiIIa}-\mathrm{SiIII}",
    # "SiIIb_SiIII": r"\mathrm{SiIIb}-\mathrm{SiIII}",
    "SiIIa_SiIII": r"\mathrm{SiIIa}",
    "SiIIb_SiIII": r"\mathrm{SiII}-\mathrm{SiIII}",
    "SiII_SiIII": r"\mathrm{SiII}-\mathrm{SiIII}",
    "SiIIc_SiIII": r"\mathrm{SiIIc}-\mathrm{SiIII}",
    "CIVa_CIVb": r"\mathrm{CIVa}-\mathrm{CIVb}",
    "MgIIa_MgIIb": r"\mathrm{MgIIa}-\mathrm{MgIIb}",
}
for metal_line in metal_lines:
    for ii in range(12):
        if (metal_line == "SiIIa_SiIII") | (metal_line == "SiIIb_SiIII"):
            param_dict["f_" + metal_line + "_" + str(ii)] = (
                r"$r_{{"
                # r"$\log f_{{"
                + metal_lines_latex[metal_line]
                + "}_"
                + str(ii)
                + "}$"
            )
        else:
            param_dict["f_" + metal_line + "_" + str(ii)] = (
                r"$f_{{"
                # r"$\log f_{{"
                + metal_lines_latex[metal_line]
                + "}_"
                + str(ii)
                + "}$"
            )
            param_dict["s_" + metal_line + "_" + str(ii)] = (
                r"$k_{{"
                # r"$\log s_{{"
                + metal_lines_latex[metal_line]
                + "}_"
                + str(ii)
                + "}$"
            )
        # param_dict["p_" + metal_line + "_" + str(ii)] = (
        #     r"$p(" + metal_lines_latex[metal_line] + "_" + str(ii) + ")$"
        # )

param_dict_rev = {v: k for k, v in param_dict.items()}

## List of all possibly free cosmology params for the truth array
## for chainconsumer plots
cosmo_params = [
    "Delta2_star",
    "n_star",
    "alpha_star",
    "f_star",
    "g_star",
    "cosmomc_theta",
    "H0",
    "mnu",
    "As",
    "ns",
    "nrun",
    "ombh2",
    "omch2",
]


## list of strings for blobs
blob_strings = [
    r"$\Delta^2_\star$",
    r"$n_\star$",
    r"$\alpha_\star$",
    r"$f_\star$",
    r"$g_\star$",
    r"$H_0$",
]
blob_strings_orig = [
    "Delta2_star",
    "n_star",
    "alpha_star",
    "f_star",
    "g_star",
    "H0",
]

conv_strings = {
    "Delta2_star": r"$\Delta^2_\star$",
    "n_star": r"$n_\star$",
    "alpha_star": r"$\alpha_\star$",
}
