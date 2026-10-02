import numpy as np

from cup1d.p1ds.base_p1d_data import BaseDataP1D


class P1D_PD2013(BaseDataP1D):
    """Represent the PD2013 P1D data product."""
    def __init__(self, z_min=0, z_max=10, use_FFT=True, add_syst=True):
        """Load the Palanque-Delabrouille et al. (2013) P1D measurement.

        Parameters
        ----------
        z_min, z_max : float, default: 0, 10
            Inclusive redshift range retained from the measurement.
        use_FFT : bool, default: True
            Read the implemented FFT table.  The alternate likelihood-table
            reader remains intentionally unimplemented.
        add_syst : bool, default: True
            Add the tabulated systematic error in quadrature to statistical
            variance before applying the published correlation matrices.
        """

        # folder storing P1D measurement
        datadir = BaseDataP1D.BASEDIR + "/PD2013/"

        # read redshifts, wavenumbers, power spectra and covariance matrices
        if use_FFT:
            z, k, Pk, cov = read_FFT_from_file(datadir, add_syst)
        else:
            z, k, Pk, cov = self.read_like_from_file(datadir, add_syst)

        super().__init__(z, k, Pk, cov, z_min=z_min, z_max=z_max)

        return


def read_FFT_from_file(datadir, add_syst=True):
    """Read the FFT-based Palanque-Delabrouille et al. (2013) P1D product.

    Parameters
    ----------
    datadir : str or path-like
        Directory containing table 4a and its per-redshift correlation tables.
    add_syst : bool, default: True
        Include systematic P1D variance in each covariance diagonal.

    Returns
    -------
    tuple
        Redshifts, common wavenumbers in ``s / km``, P1D values in ``km / s``,
        and covariance blocks in ``(km / s)**2``.
    """

    # start by reading Pk file
    p1d_file = datadir + "/table4a.dat"
    (
        iz,
        ik,
        inz,
        ink,
        inPk,
        inPkstat,
        inPknoise,
        inPkmetal,
        inPksyst,
    ) = np.loadtxt(p1d_file, unpack=True)

    # store unique values of redshift and wavenumber
    z = np.unique(inz)
    Nz = len(z)
    k = np.unique(ink)
    Nk = len(k)

    # store P1D, statistical error, noise power, metal power and systematic
    Pk = np.reshape(inPk, [Nz, Nk])
    Pkstat = np.reshape(inPkstat, [Nz, Nk])
    Pknoise = np.reshape(inPknoise, [Nz, Nk])
    Pkmetal = np.reshape(inPkmetal, [Nz, Nk])
    Pksyst = np.reshape(inPksyst, [Nz, Nk])

    # now read correlation matrices and compute covariance matrices
    cov = []
    for i in range(Nz):
        corr_file = datadir + "/cct4b" + str(i + 1) + ".dat"
        corr = np.loadtxt(corr_file, unpack=True)
        # compute variance (start with statistics only)
        var = Pkstat[i] ** 2
        if add_syst:
            var += Pksyst[i] ** 2
        sigma = np.sqrt(var)
        zcov = np.multiply(corr, np.outer(sigma, sigma))
        cov.append(zcov)

    return z, k, Pk, cov


def read_like_from_file(datadir, add_syst=True):
    """Reserve the likelihood-table reader for a future implementation.

    Parameters
    ----------
    datadir : str or path-like
        Directory containing the likelihood-table product.
    add_syst : bool, default: True
        Requested systematic-error treatment.

    Raises
    ------
    ValueError
        Always, because the likelihood-table product is not implemented.
    """

    p1d_file = datadir + "/table5a.dat"
    raise ValueError("implement _setup_like to read likelihood P1D")


def analytic_p1d_PD2013_z_kms(z, k_kms):
    """Evaluate the Palanque-Delabrouille et al. (2013) analytic P1D fit.

    Parameters
    ----------
    z : float or ndarray
        Redshift at which to evaluate the fitting formula.
    k_kms : ndarray
        Wavenumbers in ``s / km``.  Values below the model turnover are
        replaced in place by the turnover wavenumber.

    Returns
    -------
    ndarray
        P1D values in ``km / s``, flattened at low wavenumber rather than
        extrapolated to zero.
    """

    # numbers from Palanque-Delabrouille (2013)
    A_F = 0.064
    n_F = -2.55
    alpha_F = -0.1
    B_F = 3.55
    beta_F = -0.28
    k0 = 0.009
    z0 = 3.0
    n_F_z = n_F + beta_F * np.log((1 + z) / (1 + z0))
    # this function would go to 0 at low k, instead of flat power
    k_min = k0 * np.exp((-0.5 * n_F_z - 1) / alpha_F)
    flatten = k_kms < k_min
    k_kms[flatten] = k_min
    exp1 = 3 + n_F_z + alpha_F * np.log(k_kms / k0)
    toret = (
        np.pi
        * A_F
        / k0
        * pow(k_kms / k0, exp1 - 1)
        * pow((1 + z) / (1 + z0), B_F)
    )

    return toret
