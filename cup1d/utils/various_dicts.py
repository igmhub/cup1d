## Dictionary to convert likelihood parameters into latex strings
param_dict = {
    "Delta2_p": "$\Delta^2_p$",
    "mF": "$F$",
    "gamma": "$\gamma$",
    "sigT_Mpc": "$\sigma_T$",
    "kF_Mpc": "$k_F$",
    "n_p": "$n_p$",
    "Delta2_star": "$\Delta^2_\star$",
    "n_star": "$n_\star$",
    "alpha_star": "$\\alpha_\star$",
    "g_star": "$g_\star$",
    "f_star": "$f_\star$",
    "H0": "$H_0$",
    "mnu": "$\Sigma m_\\nu$",
    "As": "$A_s$",
    "ns": "$n_s$",
    "nrun": "$n_\mathrm{run}$",
    "ombh2": "$\omega_b$",
    "omch2": "$\omega_c$",
    "cosmomc_theta": "$\\theta_{MC}$",
}

for ii in range(11):
    param_dict["tau_eff_" + str(ii)] = "$\tau_{\rm eff_" + str(ii) + "}$"
    param_dict["sigT_kms_" + str(ii)] = "$\sigma_{\rm T_" + str(ii) + "}$"
    param_dict["gamma_" + str(ii)] = "$\gamma_" + str(ii) + "$"
    param_dict["kF_kms_" + str(ii)] = "$k_F_" + str(ii) + "$"
    param_dict["R_coeff_" + str(ii)] = "$R_" + str(ii) + "$"
    param_dict["ln_SN_" + str(ii)] = "$\log \mathrm{SN}_" + str(ii) + "$"
    param_dict["ln_AGN_" + str(ii)] = "$\log \mathrm{AGN}_" + str(ii) + "$"
    for jj in range(1, 5):
        param_dict["HCD_damp" + str(jj) + "_" + str(ii)] = (
            "$f_{\rm HCD" + str(jj) + "}_" + str(ii) + "$"
        )
    param_dict["HCD_const_" + str(ii)] = "$c_{\rm HCD}_" + str(ii) + "$"


metal_lines = [
    "Lya_SiIII",
    "Lya_SiII",
    "SiIIa_SiIIb",
    "SiIIa_SiIII",
    "SiIIb_SiIII",
    "CIVa_CIVb",
    "MgIIa_MgIIb",
]
metal_lines_latex = {
    "Lya_SiIII": "$\mathrm{Ly}\alpha-\mathrm{SiIII}$",
    "Lya_SiII": "$\mathrm{Ly}\alpha-\mathrm{SiII}$",
    "Lya_SiIIa": "$\mathrm{Ly}\alpha-\mathrm{SiIIa}$",
    "Lya_SiIIb": "$\mathrm{Ly}\alpha-\mathrm{SiIIb}$",
    "Lya_SiIIc": "$\mathrm{Ly}\alpha-\mathrm{SiIIc}$",
    "SiIIa_SiIIb": "$\mathrm{SiIIa}_\mathrm{SiIIb}$",
    "SiIIa_SiIII": "$\mathrm{SiIIa}_\mathrm{SiIII}$",
    "SiIIb_SiIII": "$\mathrm{SiIIb}_\mathrm{SiIII}$",
    "SiII_SiIII": "$\mathrm{SiII}_\mathrm{SiIII}$",
    "SiIIc_SiIII": "$\mathrm{SiIIc}_\mathrm{SiIII}$",
    "CIVa_CIVb": "$\mathrm{CIVa}_\mathrm{CIVb}$",
    "MgIIa_MgIIb": "$\mathrm{MgIIa}_\mathrm{MgIIb}$",
}
for metal_line in metal_lines:
    for ii in range(12):
        param_dict["f_" + metal_line + "_" + str(ii)] = (
            "$\mathrm{ln}\,f("
            + metal_lines_latex[metal_line]
            + "_"
            + str(ii)
            + ")$"
        )
        param_dict["s_" + metal_line + "_" + str(ii)] = (
            "$\mathrm{ln}\,s("
            + metal_lines_latex[metal_line]
            + "_"
            + str(ii)
            + ")$"
        )
        param_dict["p_" + metal_line + "_" + str(ii)] = (
            "$p(" + metal_lines_latex[metal_line] + "_" + str(ii) + ")$"
        )

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
    "$\Delta^2_\star$",
    "$n_\star$",
    "$\\alpha_\star$",
    "$f_\star$",
    "$g_\star$",
    "$H_0$",
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
    "Delta2_star": "$\Delta^2_\star$",
    "n_star": "$n_\star$",
    "alpha_star": "$\\alpha_\star$",
}
