import numpy as np

from cup1d.p1ds.base_p1d_data import BaseDataP1D


class P1D_Ravoux2023(BaseDataP1D):
    """Load the Ravoux et al. (2023) observed Ly-alpha P1D product.

    The reader exposes its measurements, redshifts, and covariance blocks via
    the shared :class:`BaseDataP1D` interface.
    """

    def __init__(self, z_min=0, z_max=10, velunits=True):
        """Load the Ravoux et al. (2023) P1D measurement.

        Parameters
        ----------
        z_min, z_max : float, default: 0, 10
            Inclusive redshift range retained from the measurement.
        velunits : bool, default: True
            Read the velocity-space data and covariance tables when true;
            otherwise read the published comoving-coordinate tables.
        """

        # folder storing P1D measurements
        datadir = BaseDataP1D.BASEDIR + "/Ravoux2023/"

        # read redshifts, wavenumbers, power spectra and covariance matrices
        z, k, Pk, cov = read_from_file(datadir, velunits)

        super().__init__(z, k, Pk, cov, z_min=z_min, z_max=z_max)

        return


def read_from_file(datadir, velunits):
    """Read Ravoux et al. (2023) P1D values and covariance tables.

    Parameters
    ----------
    datadir : str or path-like
        Directory containing the measurement and covariance text files.
    velunits : bool
        Select velocity-space or comoving-coordinate products.

    Returns
    -------
    tuple
        Redshift bins, common wavenumber grid, P1D values, and covariance
        blocks.  For ``velunits=True``, the units are ``s / km``, ``km / s``,
        and ``(km / s)**2``, respectively.
    """

    # start by reading Pk file
    if velunits:
        p1d_file = datadir + "/p1d_measurement_kms.txt"
    else:
        p1d_file = datadir + "/p1d_measurement.txt"

    inz, ink, inPk = np.loadtxt(
        p1d_file,
        unpack=True,
        usecols=range(
            3,
        ),
    )
    # store unique values of redshift and wavenumber
    z = np.unique(inz)
    Nz = len(z)

    mask = inz == z[0]
    k = ink[mask]
    Nk = len(k)

    # re-shape matrices, and compute variance (statistics only for now)
    if velunits:
        Pk = []
        for i in range(len(z)):
            mask = inz == z[i]
            Pk.append(inPk[mask][:Nk] * np.pi / k)
        Pk = np.array(Pk)
    else:
        Pk = np.reshape(inPk * np.pi / k, [Nz, Nk])

    # now read correlation matrices
    if velunits:
        cov_file = datadir + "/covariance_matrix_kms.txt"
    else:
        cov_file = datadir + "/covariance_matrix.txt"

    inzcov, ink1, _, incov = np.loadtxt(
        cov_file,
        unpack=True,
        usecols=range(
            4,
        ),
    )
    if velunits:
        cov_Pk = []
        for i in range(Nz):
            mask = inzcov == z[i]
            k1 = np.unique(ink1[mask])
            cov_Pk_z = []
            for j in range(Nk):
                mask_k = mask & (ink1 == k1[j])
                cov_Pk_z.append(incov[mask_k][:Nk])
            cov_Pk.append(cov_Pk_z)
        cov_Pk = np.array(cov_Pk)
    else:
        cov_Pk = np.reshape(
            incov,
            [Nz, Nk, Nk],
        )

    return z, k, Pk, cov_Pk
