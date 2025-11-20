import os
import math
import numpy as np
from scipy.stats import chi2 as chi2_scipy


def match_precision(x, xp, xm, sig=2):
    """
    Return LaTeX string "$x^{+xp}_{-xm}$" with x and errors rounded so that
    errors have `sig` significant figures.
    If x is positive, add LaTeX thin space prefix for alignment: '\;\;\,'.
    """
    err = max(abs(xp), abs(xm))
    if err == 0:
        s = f"${x:.3f}$"
    else:
        exp_err = int(math.floor(math.log10(err)))
        digits = sig - exp_err - 1
        digits = max(digits, 0)
        fmt = f"{{:.{digits}f}}"
        x_s = fmt.format(x)
        xp_s = fmt.format(xp)
        xm_s = fmt.format(xm)
        s = f"${x_s}^{{+{xp_s}}}_{{-{xm_s}}}$"

    if x >= 0:
        s = r"$\;\;\," + s[1:]  # prepend LaTeX thin space, keep $...$
    return s


def format_last(val):
    """Scientific notation if |val| < 1e-3 (and val != 0), else 4 decimals."""
    if val == 0:
        return "0.00"
    if abs(val) < 1e-2:
        coeff, exp = f"{val:.1e}".split("e")
        exp = int(exp)
        return f"${coeff}\\times10^{{{exp}}}$"
    return f"{val:.2f}"


def make_latex_table(
    table, color_threshold=[0.9655, 2.2957], colors=["yellow", "red"]
):
    """
    Print aligned LaTeX rows from `table`.
    Each row: [name, x1, x1p, x1m, x2, x2p, x2m, val3, val4, val5]
    - columns 2 & 3: $value^{+err}_{-err}$; positive values get '\;\;\,' padding
    - column 4 (val3) -> formatted as .2f, triggers coloring if < color_threshold
    - column 5 (val4) -> formatted as .1f
    - column 6 (val5) -> .4f or scientific if <1e-3
    """
    rows_plain = []
    for row in table:
        name = str(row[0])
        x1, x1p, x1m = row[1:4]
        x2, x2p, x2m = row[4:7]
        val3, val4, val5 = row[7:]

        p1 = match_precision(x1, x1p, x1m, sig=2)
        p2 = match_precision(x2, x2p, x2m, sig=2)
        s3 = f"{val3:.2f}"
        s4 = f"{val4:.1f}"
        s5 = format_last(val5)

        # store val3 for threshold checking
        rows_plain.append([name, p1, p2, s3, s4, s5, val3])

    # compute column widths
    ncols = 6
    col_widths = [0] * ncols
    for r in rows_plain:
        for i in range(ncols):
            col_widths[i] = max(col_widths[i], len(str(r[i])))

    for r in rows_plain:
        name, p1, p2, s3, s4, s5, val3num = r

        # decide if coloring is needed
        if float(val3num) > color_threshold[0]:
            colorize = True
            if float(val3num) > color_threshold[1]:
                color = colors[1]
            else:
                color = colors[0]
        else:
            colorize = False
        colorize = False

        if colorize:
            # no padding
            cells = [
                f"\\textcolor{{{color}}}{{{name}}}",
                f"\\textcolor{{{color}}}{{{p1}}}",
                f"\\textcolor{{{color}}}{{{p2}}}",
                f"\\textcolor{{{color}}}{{{s3}}}",
                f"\\textcolor{{{color}}}{{{s4}}}",
                f"\\textcolor{{{color}}}{{{s5}}}",
            ]
        else:
            # pad for neat alignment
            cells = [
                str(name).ljust(col_widths[0]),
                p1.ljust(col_widths[1]),
                p2.ljust(col_widths[2]),
                s3.ljust(col_widths[3]),
                s4.ljust(col_widths[4]),
                s5.ljust(col_widths[5]),
            ]

        line = " & ".join(cells) + " \\\\"
        print(line)


def format_last_column(values):
    """Format last column with trailing zeros or LaTeX scientific notation."""
    formatted = []
    for val in values:
        if abs(val) >= 1e-3:
            s = f"{val:.4f}"  # fixed 4 decimals
        else:
            coeff, exp = f"{val:.1e}".split("e")
            exp = int(exp)
            s = f"${coeff}\\times10^{{{exp}}}$"
        formatted.append(s)
    width = max(len(s) for s in formatted)
    return [f"{s:>{width}}" for s in formatted]


def format_column(
    values,
    sigfigs=2,
    force_decimals=True,
    one_decimal=False,
    two_decimals=False,
):
    formatted = []
    for val in values:
        if one_decimal:
            s = f"{val:.1f}"
        elif two_decimals:
            s = f"{val:.2f}"
        elif force_decimals:
            s = f"{val:.3f}"
        else:
            s = f"{val:.{sigfigs}g}"
        formatted.append(s)
    width = max(len(s) for s in formatted)
    return [f"{s:>{width}}" for s in formatted]


def table_variations(base):
    variations = {
        "DESIY1_QMLE3_mpg": [
            "Fiducial",
            "DESIY1_QMLE3/global_opt/CH24_mpgcen_gpr/chain_7/",
        ],
        "zmax": [
            "Data: $z \\leq 3.4$",
            "DESIY1_QMLE3/zmax/CH24_mpgcen_gpr/chain_2/",
        ],
        "zmin": [
            "Data: $z \\geq 2.6$",
            "DESIY1_QMLE3/zmin/CH24_mpgcen_gpr/chain_2/",
        ],
        "DESIY1_QMLE_mpg": [
            "Data: w/ low SNR",
            "DESIY1_QMLE/global_opt/CH24_mpgcen_gpr/chain_2/",
        ],
        "DESIY1_FFT3_dir_mpg": [
            "Data: FFT",
            "DESIY1_FFT3_dir/global_opt/CH24_mpgcen_gpr/chain_2/",
        ],
        # "DESIY1_FFT_dir_mpg": ["Data: FFT w/ low SNR", "DESIY1_FFT_dir/global_opt/CH24_mpgcen_gpr/chain_1/"],
        "dat_syst_diag": [
            "Cov: uncorr syst",
            "DESIY1_QMLE3/data_syst_diag/CH24_mpgcen_gpr/chain_2/",
        ],
        "no_inflate": [
            "Cov: w/o 5\% err",
            "DESIY1_QMLE3/no_inflate/CH24_mpgcen_gpr/chain_2/",
        ],
        "no_emu_cov": [
            "Cov: w/o emu err",
            "DESIY1_QMLE3/no_emu_cov/CH24_mpgcen_gpr/chain_2/",
        ],
        "emu_diag": [
            "Cov: emu diag",
            "DESIY1_QMLE3/emu_diag/CH24_mpgcen_gpr/chain_3/",
        ],
        "emu_block": [
            "Cov: emu block diag",
            "DESIY1_QMLE3/emu_block/CH24_mpgcen_gpr/chain_3/",
        ],
        "DESIY1_QMLE3_nyx": [
            "Emulator: lace-lyssa",
            "DESIY1_QMLE3/global_opt/CH24_nyxcen_gpr/chain_3/",
        ],
        "cosmo": [
            "Cosmo: $\omega_0\omega_a$CDM",
            "DESIY1_QMLE3/cosmo/CH24_mpgcen_gpr/chain_3/",
        ],
        "cosmo_h74": [
            "Cosmo: $h=0.74$",
            "DESIY1_QMLE3/cosmo_h74/CH24_mpgcen_gpr/chain_1/",
        ],
        "cosmo_mnu": [
            r"Cosmo: $\sum m_\nu=0.3$ eV",
            "DESIY1_QMLE3/cosmo_mnu/CH24_mpgcen_gpr/chain_2/",
        ],
        "cosmo_high": [
            "Cosmo: high $\Omega_\mathrm{cdm}h^2$",
            "DESIY1_QMLE3/cosmo_high/CH24_mpgcen_gpr/chain_2/",
        ],
        "cosmo_low": [
            "Cosmo: low $\Omega_\mathrm{cdm}h^2$",
            "DESIY1_QMLE3/cosmo_low/CH24_mpgcen_gpr/chain_2/",
        ],
        "more_igm": [
            "IGM: $n_z=8$",
            "DESIY1_QMLE3/more_igm/CH24_mpgcen_gpr/chain_2/",
        ],
        "hcd0": [
            "HCD: w/ $f_{\\mathrm{const}}^{\\mathrm{HCD}}$",
            "DESIY1_QMLE3/HCD0/CH24_mpgcen_gpr/chain_2/",
        ],
        "dlas": [
            "HCD: only DLAs",
            "DESIY1_QMLE3/DLAs/CH24_mpgcen_gpr/chain_2/",
        ],
        "HCD_BOSS": [
            "HCD: BOSS",
            "DESIY1_QMLE3/HCD_BOSS/CH24_mpgcen_gpr/chain_2/",
        ],
        "metal_thin": [
            "Metals: opt thin",
            "DESIY1_QMLE3/metal_thin/CH24_mpgcen_gpr/chain_2/",
        ],
        "metal_deco": [
            "Metals: no H-Si decorr",
            "DESIY1_QMLE3/metal_deco/CH24_mpgcen_gpr/chain_2/",
        ],
        "metal_si2": [
            "Metals: no SiII-SiII",
            "DESIY1_QMLE3/metal_si2/CH24_mpgcen_gpr/chain_2/",
        ],
        "metal_trad": [
            "Metals: BOSS",
            "DESIY1_QMLE3/metal_trad/CH24_mpgcen_gpr/chain_2/",
        ],
        "Metals_Ma2025": [
            "Metals: Ma+2025",
            "DESIY1_QMLE3/Metals_Ma2025/CH24_mpgcen_gpr/chain_2/",
        ],
    }

    fid_vals = {}

    table = []
    blobs = np.load(
        os.path.join(base, variations["DESIY1_QMLE3_mpg"][1], "blobs.npy")
    )
    corr_fid = np.corrcoef(
        blobs["Delta2_star"].reshape(-1), blobs["n_star"].reshape(-1)
    )[1, 0]
    blobs = 0

    for ii, var in enumerate(variations):
        folder = os.path.join(base, variations[var][1])
        # print(var, np.load(folder + "lnprob.npy").shape)
        # print(var,variations[var][1])
        # plots_chain(folder)
        data = np.load(
            os.path.join(folder, "summary.npy"), allow_pickle=True
        ).item()
        delta2_star = data["delta2_star_16_50_84"][1]
        n_star = data["n_star_16_50_84"][1]
        if ii == 0:
            fid_vals["delta2_star"] = delta2_star
            fid_vals["delta2_star_2dcen"] = data["delta2_star_2dcen"]
            fid_vals["delta2_star_err"] = data["delta2_star_err"]
            fid_vals["n_star"] = n_star
            fid_vals["nstar_2dcen"] = data["nstar_2dcen"]
            fid_vals["n_star_err"] = data["n_star_err"]

        row = []
        row.append(variations[var][0])
        row.append(delta2_star - fid_vals["delta2_star"])
        top = data["delta2_star_16_50_84"][2] - data["delta2_star_16_50_84"][1]
        bot = data["delta2_star_16_50_84"][1] - data["delta2_star_16_50_84"][0]
        row.append(top)
        row.append(bot)
        row.append(n_star - fid_vals["n_star"])
        top = data["n_star_16_50_84"][2] - data["n_star_16_50_84"][1]
        bot = data["n_star_16_50_84"][1] - data["n_star_16_50_84"][0]
        row.append(top)
        row.append(bot)

        diffx = data["delta2_star_2dcen"] - fid_vals["delta2_star_2dcen"]
        errx = np.max([data["delta2_star_err"], fid_vals["delta2_star_err"]])
        diffy = data["nstar_2dcen"] - fid_vals["nstar_2dcen"]
        erry = np.max([data["n_star_err"], fid_vals["n_star_err"]])
        consist = (
            1
            / (1 - corr_fid**2)
            * (
                diffx**2 / errx**2
                - 2 * corr_fid * diffx * diffy / errx / erry
                + diffy**2 / erry**2
            )
        )
        # print(var, consist)
        # row.append(chi2_scipy.sf(consist, 2))
        row.append(consist)

        row.append(data["chi2"])
        row.append(data["prob_chi2"])

        table.append(row)

    make_latex_table(table)

    return
