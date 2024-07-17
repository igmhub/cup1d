import os
import numpy as np
import pandas

from cup1d.p1ds.base_p1d_data import BaseDataP1D


class P1D_QMLE_Ohio(BaseDataP1D):
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
        """Read measured P1D from file from Ohio mocks (QMLE)

        Args:
            filename: if not None, read that file.
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
        """Read file containing mock P1D"""

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
