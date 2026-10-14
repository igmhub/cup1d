import numpy as np

from cup1d.p1ds.base_p1d_data import BaseDataP1D


class P1D_Irsic2017(BaseDataP1D):
    """Load the Irsic et al. (2017) observed Ly-alpha P1D product.

    Measurements and covariance blocks are normalized to cup1d's native
    velocity-space binned-data interface.
    """

    def __init__(self, z_min=0, z_max=10, add_syst=True, ignore_zcov=True):
        """Load the Iršič et al. (2017) P1D measurement.

        Parameters
        ----------
        z_min, z_max : float, default: 0, 10
            Inclusive redshift interval retained from the measurement.
        add_syst : bool, default: True
            Add tabulated systematic variances to each redshift-block diagonal.
        ignore_zcov : bool, default: True
            Require the currently implemented block-diagonal treatment.  Full
            cross-redshift covariance is not yet supported.
        """

        # folder storing P1D measurement
        datadir = BaseDataP1D.BASEDIR + "/Irsic2017/"

        z, k_kms, Pk_kms, cov_Pk_kms = read_from_file(datadir, add_syst, ignore_zcov)

        super().__init__(z, k_kms, Pk_kms, cov_Pk_kms, z_min=z_min, z_max=z_max)

        return


def read_from_file(basedir, add_syst, ignore_zcov):
    """Read Iršič et al. (2017) P1D tables and per-redshift covariance blocks.

    Parameters
    ----------
    basedir : str or path-like
        Directory containing the published power and covariance text files.
    add_syst : bool
        Add the tabulated systematic P1D errors in quadrature.
    ignore_zcov : bool
        Must be true because this reader exposes only within-redshift blocks.

    Returns
    -------
    tuple
        Redshifts, common wavenumbers in ``s / km``, P1D values in ``km / s``,
        and covariance blocks in ``(km / s)**2``.

    Raises
    ------
    AssertionError
        If cross-redshift covariance is requested.
    """

    assert ignore_zcov, "implement cross-z covariance in p1d_Irsic2017"

    p1d_file = basedir + "/pk_xs_final.txt"
    inz, ink, inPk, inPkstat, inPksyst, _, _ = np.loadtxt(p1d_file, unpack=True)
    # store unique values of redshift and wavenumber
    z = np.unique(inz)
    Nz = len(z)
    k_kms = np.unique(ink)
    Nk = len(k_kms)

    # store P1D, statistical error, noise power, metal power and systematic
    Pk_kms = np.reshape(inPk, [Nz, Nk])
    Pkstat = np.reshape(inPkstat, [Nz, Nk])
    Pksyst = np.reshape(inPksyst, [Nz, Nk])

    # read covariance with statistical uncertainty
    cov_file = basedir + "/cov_pk_xs_final.txt"
    _, _, inCov = np.loadtxt(cov_file, unpack=True)
    cov_syst = inCov.reshape(Nz * Nk, Nz * Nk)

    # TBD Add full covariance
    # for now use diagonal covariance matrices
    cov_Pk_kms = []
    for iz in range(Nz):
        # get covariance for z bin only
        zcov = cov_syst[iz * Nk : (iz + 1) * Nk, iz * Nk : (iz + 1) * Nk]
        if add_syst:
            zcov += np.diag(Pksyst[iz] ** 2)
        cov_Pk_kms.append(zcov)

    return z, k_kms, Pk_kms, cov_Pk_kms
