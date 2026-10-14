

def gaussian_chi2(neff, DL2, neff_val, DL2_val, neff_err, DL2_err, r):
    """Evaluate a correlated Gaussian compressed linear-power chi-squared.

    Parameters
    ----------
    neff, DL2 : float or array-like
        Effective slope and dimensionless linear-power amplitude at the
        experiment's stated velocity-space pivot.
    neff_val, DL2_val : float
        Measured central values.
    neff_err, DL2_err : float
        One-standard-deviation marginal errors.
    r : float
        Correlation coefficient between slope and amplitude.

    Returns
    -------
    float or ndarray
        Delta chi-squared relative to the measurement central value.
    """
    chi2 = (
        (DL2 - DL2_val) ** 2 / DL2_err**2
        + (neff - neff_val) ** 2 / neff_err**2
        - 2 * r * (neff - neff_val) * (DL2 - DL2_val) / DL2_err / neff_err
    ) / (1 - r * r)
    return chi2


def gaussian_chi2_McDonald2005(neff, DL2):
    """Evaluate the McDonald et al. (2005) compressed linear-power likelihood.

    ``neff`` and dimensionless ``DL2`` are defined at ``z=3`` and
    ``k=0.009 s/km``. The returned dictionary contains the adopted Gaussian
    summary and the evaluated chi-squared.
    """
    # DL2 = k^3 P(k) / (2 pi^2) , at z=3
    DL2_val = 0.47
    DL2_err = 0.06
    # neff = effective slope at kp = 0.009 s/km, i.e., d ln P / dln k
    neff_val = -2.3
    neff_err = 0.055
    # correlation coefficient
    r = 0.6
    results = {
        "Delta2_star": DL2_val,
        "Delta2_star_err": DL2_err,
        "n_star": neff_val,
        "n_star_err": neff_err,
        "r": r,
        "chi2": gaussian_chi2(
            neff, DL2, neff_val, DL2_val, neff_err, DL2_err, r
        ),
    }
    return results


def gaussian_chi2_Chabanier2019(neff, DL2):
    """Evaluate the Chabanier et al. (2019) compressed likelihood at z=3.

    The returned dictionary includes the adopted published Gaussian summary
    and chi-squared for ``neff`` and dimensionless ``DL2`` at ``0.009 s/km``.
    """
    # DL2 = k^3 P(k) / (2 pi^2), at z=3
    DL2_val = 0.310
    DL2_err = 0.020
    # neff = effective slope at kp = 0.009 s/km, i.e., d ln P / dln k
    neff_val = -2.340
    neff_err = 0.0060
    # correlation coefficient
    r = 0.512
    results = {
        "Delta2_star": DL2_val,
        "Delta2_star_err": DL2_err,
        "n_star": neff_val,
        "n_star_err": neff_err,
        "r": r,
        "chi2": gaussian_chi2(
            neff, DL2, neff_val, DL2_val, neff_err, DL2_err, r
        ),
    }
    return results


def gaussian_chi2_PalanqueDelabrouille2015(neff, DL2):
    """Evaluate the Palanque-Delabrouille et al. (2015) Gaussian summary.

    Parameters and returned fields follow :func:`gaussian_chi2_McDonald2005`.
    """
    # DL2 = k^3 P(k) / (2 pi^2), at z=3
    DL2_val = 0.32
    DL2_err = 0.03
    # neff = effective slope at kp = 0.009 s/km, i.e., d ln P / dln k
    neff_val = -2.36
    neff_err = 0.01
    # correlation coefficient (no idea from where)
    r = 0.55
    results = {
        "Delta2_star": DL2_val,
        "Delta2_star_err": DL2_err,
        "n_star": neff_val,
        "n_star_err": neff_err,
        "r": r,
        "chi2": gaussian_chi2(
            neff, DL2, neff_val, DL2_val, neff_err, DL2_err, r
        ),
    }
    return results


def gaussian_chi2_Walther2024(neff, DL2, ana_type="priors"):
    """Evaluate one of the Walther et al. (2024) compressed summaries.

    Parameters
    ----------
    neff, DL2 : float or array-like
        Linear-power slope and dimensionless amplitude at the stated pivot.
    ana_type : str, default="priors"
        Select the prior-including published summary; another value selects
        the no-prior summary.

    Returns
    -------
    dict
        Adopted summary statistics and evaluated chi-squared.
    """

    if ana_type == "priors":
        # DL2 = k^3 P(k) / (2 pi^2) , at z=3
        DL2_val = 0.388
        DL2_err = 0.045
        # neff = effective slope at kp = 0.009 s/km, i.e., d ln P / dln k
        neff_val = -2.2978
        neff_err = 0.0067
        # correlation coefficient
        r = 0.632
        print("using prior")
    else:
        # DL2 = k^3 P(k) / (2 pi^2), at z=3
        DL2_val = 0.260
        DL2_err = 0.024
        # neff = effective slope at kp = 0.009 s/km, i.e., d ln P / dln k
        neff_val = -2.2995
        neff_err = 0.0066
        # correlation coefficient
        r = 0.161
        print("no prior")

    results = {
        "Delta2_star": DL2_val,
        "Delta2_star_err": DL2_err,
        "n_star": neff_val,
        "n_star_err": neff_err,
        "r": r,
        "chi2": gaussian_chi2(
            neff, DL2, neff_val, DL2_val, neff_err, DL2_err, r
        ),
    }
    return results


def gaussian_chi2_DESI_DR1(neff, DL2):
    """Compute the Gaussian DESI DR1 compressed-likelihood chi-squared.

    The likelihood is defined at ``z_star = 3.0`` and
    ``k_star_kms = 0.009`` using the published amplitude--slope covariance.

    Parameters
    ----------
    neff, DL2 : float or array-like
        Effective slope and dimensionless amplitude at the DR1 pivot.

    Returns
    -------
    dict
        Published Gaussian summary statistics and evaluated chi-squared.
    """
    # DL2 = k^3 P(k) / (2 pi^2), at z=3
    DL2_val = 0.379
    DL2_err = 0.032
    # neff = effective slope at kp = 0.009 s/km, i.e., d ln P / dln k
    neff_val = -2.309
    neff_err = 0.019
    # correlation coefficient
    r = -0.1738
    results = {
        "Delta2_star": DL2_val,
        "Delta2_star_err": DL2_err,
        "n_star": neff_val,
        "n_star_err": neff_err,
        "r": r,
        "chi2": gaussian_chi2(
            neff, DL2, neff_val, DL2_val, neff_err, DL2_err, r
        ),
    }
    return results


# Kept while downstream notebooks migrate to the dataset-based name.
gaussian_chi2_ChavesMontero2026 = gaussian_chi2_DESI_DR1
