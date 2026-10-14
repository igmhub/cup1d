import os
import numpy as np
import pandas

from cup1d.p1ds.base_p1d_data import BaseDataP1D


class P1D_QMLE_Ohio(BaseDataP1D):
    """Represent the QMLE Ohio P1D data product."""
    def __init__(
        self,
        diag_cov=True,
        kmin_kms=0.001,
        kmax_kms=0.04,
        z_min=0,
        z_max=10,
        version="ohio-v0",
        filename=None,
        noise_syst=0,
    ):
        """Load a QMLE Ohio mock P1D data product.

        Parameters
        ----------
        diag_cov : bool, default: True
            Retained for API compatibility.  This reader currently constructs
            a diagonal covariance, plus an optional rank-one noise systematic.
        kmin_kms, kmax_kms : float, default: 0.001, 0.04
            Strict wavenumber cuts in ``s / km``.
        z_min, z_max : float, default: 0, 10
            Inclusive redshift selection applied by :class:`BaseDataP1D`.
        version : str, default: "ohio-v0"
            Distributed mock-product version used when ``filename`` is absent.
        filename : str or path-like, optional
            Explicit QMLE table.  Overrides the ``P1D_FORECAST`` lookup.
        noise_syst : float, default: 0
            Amplitude of a fully correlated noise systematic, expressed as a
            multiplier of the table's ``b`` column.
        """

        # read redshifts, wavenumbers, power spectra and covariance matrices
        z, k, Pk, cov = self._read_file(
            diag_cov, kmin_kms, kmax_kms, version, filename, noise_syst
        )

        super().__init__(z, k, Pk, cov, z_min=z_min, z_max=z_max)

        return

    def _read_file(
        self, diag_cov, kmin_kms, kmax_kms, version, filename, noise_syst
    ):
        """Read and scale-cut the QMLE Ohio mock table.

        Parameters
        ----------
        diag_cov : bool
            API-compatible covariance selection flag; diagonal covariance is
            currently always used.
        kmin_kms, kmax_kms : float
            Strict retained wavenumber limits in ``s / km``.
        version : str
            Built-in product version selected when ``filename`` is omitted.
        filename : str or path-like, optional
            Explicit input table.
        noise_syst : float
            Multiplier for the fully correlated noise-systematic component.

        Returns
        -------
        tuple
            Redshift bins, per-redshift wavenumbers in ``s / km``, P1D values
            in ``km / s``, and covariance blocks in ``(km / s)**2``.

        Raises
        ------
        ValueError
            If a requested built-in mock version is unknown.
        AssertionError
            If the required ``P1D_FORECAST`` environment variable or input
            table is unavailable.
        """

        if filename:
            fname = filename
        else:
            # DESI members can access this data in GitHub (cosmodesi/p1d_forecast)
            assert "P1D_FORECAST" in os.environ, "Define P1D_FORECAST variable"
            basedir = (
                os.environ["P1D_FORECAST"] + "/private_data/p1d_measurements/"
            )
            datadir = basedir + "/QMLE_Ohio/"

            # for now we can only handle diagonal covariances
            if version == "ohio-v0":
                fname = (
                    datadir
                    + "/desi-y5fp-1.5-4-o3-deconv-power-qmle_kmax0.04.txt"
                )
            else:
                raise ValueError("unknown version of DESI P1D " + version)

        # start by reading the file with measured band power
        print("will read P1D file", fname)
        assert os.path.isfile(fname), "Ask Naim for P1D file"

        data = pandas.read_table(
            fname, comment="#", delim_whitespace=True
        ).to_records(index=False)
        # z k1 k2 kc Pfid ThetaP Pest ErrorP d b t
        zbins = np.unique(data["z"])
        Nz = zbins.shape[0]

        k = []
        Pk = []
        cov = []

        for z in zbins:
            mask = np.argwhere(
                (data["z"] == z)
                & (data["kc"] > kmin_kms)
                & (data["kc"] < kmax_kms)
            )[:, 0]

            k.append(data["kc"][mask])
            Pk.append(data["Pest"][mask])

            var = data["ErrorP"][mask] ** 2
            C = np.diag(var)
            if noise_syst > 0:
                pnoise = noise_syst * data["b"][mask]
                C += np.outer(pnoise, pnoise)

            cov.append(C)

        return zbins, k, Pk, cov
