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


def make_latex_table(table, color_threshold=0.25, color="red"):
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
        colorize = float(val3num) < color_threshold

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
            "DESIY1_QMLE3/global_opt/CH24_mpgcen_gpr/chain_1/",
        ],
        "zmax": [
            "Data: $z \\leq 3.4$",
            "DESIY1_QMLE3/zmax/CH24_mpgcen_gpr/chain_1/",
        ],
        "zmin": [
            "Data: $z \\geq 2.6$",
            "DESIY1_QMLE3/zmin/CH24_mpgcen_gpr/chain_1/",
        ],
        "DESIY1_QMLE_mpg": [
            "Data: w/ low SNR",
            "DESIY1_QMLE/global_opt/CH24_mpgcen_gpr/chain_1/",
        ],
        "DESIY1_FFT3_dir_mpg": [
            "Data: FFT",
            "DESIY1_FFT3_dir/global_opt/CH24_mpgcen_gpr/chain_1/",
        ],
        # "DESIY1_FFT_dir_mpg": ["Data: FFT w/ low SNR", "DESIY1_FFT_dir/global_opt/CH24_mpgcen_gpr/chain_1/"],
        "no_emu_cov": [
            "Cov: w/o emu err",
            "DESIY1_QMLE3/no_emu_cov/CH24_mpgcen_gpr/chain_1/",
        ],
        "no_inflate": [
            "Cov: w/o 5\% err",
            "DESIY1_QMLE3/no_inflate/CH24_mpgcen_gpr/chain_1/",
        ],
        "no_inflate_no_emu_cov": [
            "Cov: w/o emu, 5\% err",
            "DESIY1_QMLE3/no_inflate_no_emu_cov/CH24_mpgcen_gpr/chain_1/",
        ],
        "DESIY1_QMLE3_nyx": [
            "Model: lace-lyssa",
            "DESIY1_QMLE3/global_opt/CH24_nyxcen_gpr/chain_1/",
        ],
        "cosmo": [
            "Model: $\omega_0\omega_a$CDM",
            "DESIY1_QMLE3/cosmo/CH24_mpgcen_gpr/chain_1/",
        ],
        "cosmo_high": [
            "Model: high $\Omega_\mathrm{M}h^2$",
            "DESIY1_QMLE3/cosmo_high/CH24_mpgcen_gpr/chain_1/",
        ],
        "cosmo_low": [
            "Model: low $\Omega_\mathrm{M}h^2$",
            "DESIY1_QMLE3/cosmo_low/CH24_mpgcen_gpr/chain_1/",
        ],
        "more_igm": [
            "Model: IGM $n_z=8$",
            "DESIY1_QMLE3/more_igm/CH24_mpgcen_gpr/chain_1/",
        ],
        "less_igm": [
            "Model: IGM $n_z=4$",
            "DESIY1_QMLE3/less_igm/CH24_mpgcen_gpr/chain_1/",
        ],
        "Turner24": [
            "Model: $\\bar{F}\\, n_z=1$",
            "DESIY1_QMLE3/Turner24/CH24_mpgcen_gpr/chain_1/",
        ],
        "hcd_z": [
            "Model: HCD $n_z=2$",
            "DESIY1_QMLE3/hcd_z/CH24_mpgcen_gpr/chain_1/",
        ],
        "dlas": [
            "Model: only DLAs",
            "DESIY1_QMLE3/DLAs/CH24_mpgcen_gpr/chain_1/",
        ],
        "metals_z": [
            "Model: metals $n_z=2$",
            "DESIY1_QMLE3/metals_z/CH24_mpgcen_gpr/chain_1/",
        ],
        "metal_trad": [
            "Model: trad metal",
            "DESIY1_QMLE3/metal_trad/CH24_mpgcen_gpr/chain_1/",
        ],
        "metal_thin": [
            "Model: metal thin",
            "DESIY1_QMLE3/metal_thin/CH24_mpgcen_gpr/chain_1/",
        ],
        "metal_deco": [
            "Model: no metal decorr",
            "DESIY1_QMLE3/metal_deco/CH24_mpgcen_gpr/chain_1/",
        ],
        "metal_si2": [
            "Model: no SiII-SiII",
            "DESIY1_QMLE3/metal_si2/CH24_mpgcen_gpr/chain_1/",
        ],
        "no_res": [
            "Model: no resolution",
            "DESIY1_QMLE3/no_res/CH24_mpgcen_gpr/chain_1/",
        ],
    }

    fid_vals = {}

    table = []
    # blobs = np.load(os.path.join(base, variations["DESIY1_QMLE3_mpg"][1], "blobs.npy"))
    # corr_fid= np.corrcoef(blobs["Delta2_star"].reshape(-1), blobs["n_star"].reshape(-1))[1,0]
    # blobs = 0
    corr_fid = 0.186

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
        row.append(chi2_scipy.sf(consist, 2))

        row.append(data["chi2"])
        row.append(data["prob_chi2"])

        table.append(row)

    make_latex_table(table)

    return
