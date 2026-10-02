
import pandas
import numpy as np

from cup1d.p1ds.base_p1d_data import BaseDataP1D


class P1D_Karacayli2024(BaseDataP1D):
    """Represent the Karacayli2024 P1D data product."""
    def __init__(self, diag_cov=False, kmax_nyq=0.5, z_min=2.19, z_max=10):
        """Load the Karacayli et al. (2024) DESI early-data P1D measurement.

        Parameters
        ----------
        diag_cov : bool, default: False
            Use only published total variances instead of the off-diagonal
            covariance product.
        kmax_nyq : float, default: 0.5
            Fraction of the redshift-dependent Nyquist wavenumber retained.
        z_min, z_max : float, default: 2.19, 10
            Inclusive redshift selection.  The default excludes the
            unrecommended lowest-redshift bin.
        """

        # read redshifts, wavenumbers, power spectra and covariance matrices
        z, k, Pk, cov = read_from_file(diag_cov, kmax_nyq)

        super().__init__(z, k, Pk, cov, z_min=z_min, z_max=z_max)

        return


def read_from_file(diag_cov, kmax_nyq):
    """Read the Karacayli et al. (2024) P1D and apply a Nyquist-scale cut.

    Parameters
    ----------
    diag_cov : bool
        Construct a diagonal covariance from total errors when true; otherwise
        read the published total off-diagonal covariance.
    kmax_nyq : float
        Fraction of each redshift bin's Nyquist frequency to retain.

    Returns
    -------
    tuple
        Redshift bins, common wavenumbers in ``s / km``, padded P1D vectors in
        ``km / s``, and covariance blocks in ``(km / s)**2``.  Padded modes
        receive zero power and infinite variance for later removal.
    """

    # folder storing P1D measurement
    datadir = BaseDataP1D.BASEDIR + "/Karacayli2024/"
    fname = datadir + "/desi-edrp-lyasb1subt-p1d-detailed-results.txt"

    with open(fname) as _:
        names = _.readline()[1:].strip().split()

    # start by reading the file with measured band power
    # z k1 k2 kc Pfid ThetaP p_final e_stat p_raw p_noise p_fid_qmle p_smooth
    # e_n_syst e_res_syst e_cont_syst e_dla_syst p_sb1 e_sb1_stat e_total
    data = pandas.read_table(
        fname, comment="#", names=names, delim_whitespace=True
    ).to_records(index=False)

    if diag_cov:
        cov_full = np.diag(data["e_total"] ** 2)
    else:
        fname_cov = (
            datadir + "/desi-edrp-lyasb1subt-cov-total-offdiag-results.txt"
        )
        cov_full = np.loadtxt(fname_cov)

    zbins = np.unique(data["z"])
    kbins = np.unique(data["kc"])
    Nk = kbins.size
    Nz = zbins.size

    print("Nz = {} , Nk = {}".format(Nz, Nk))
    Pk = []
    cov = []
    for iz in range(Nz):
        z = zbins[iz]
        # moving Nyquist frequency of the DESI wavelength
        # grid. dlambda = 0.8 A
        dv = 2.99792458e5 * 0.8 / 1215.67 / (1 + z)
        kmax = kmax_nyq * np.pi / dv

        w = np.isclose(data["z"], z) & (data["kc"] < kmax)
        tmp_d = np.zeros(Nk)
        _nk = w.sum()
        tmp_d[:_nk] = data["p_final"][w]
        Pk.append(tmp_d)

        tmp_cov = np.zeros((Nk, Nk))
        # Fill non-existing k bins with inf
        # such that covariance is still invertible
        np.fill_diagonal(tmp_cov, np.inf)
        tmp_cov[:_nk, :_nk] = cov_full[w, :][:, w]
        cov.append(tmp_cov)

    return zbins, kbins, Pk, cov
