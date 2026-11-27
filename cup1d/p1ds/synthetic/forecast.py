"""Class to generate a mock P1D from another P1D object and an emulator"""

import numpy as np
from cup1d.p1ds.synthetic.base_p1d_mock import BaseMockP1D

from cup1d.p1ds.observations import (
    data_Chabanier2019,
    data_DESIY1,
)


class Forecast_P1D(BaseMockP1D):
    """Mock P1D data with an observed survey's grids and covariance structure.

    The class loads a reference data product for its redshift, wavenumber, and
    covariance blocks, then replaces its P1D central values with a prediction
    from the supplied theory.
    """

    def __init__(
        self,
        theory,
        data_label="Chabanier2019",
        z_min=0,
        z_max=10,
        add_noise=False,
        seed=0,
    ):
        """Build a forecast from a theory prediction and reference data layout.

        Parameters
        ----------
        theory : object
            Initialized Cup1D theory object used to evaluate the fiducial P1D.
        data_label : str, default: "Chabanier2019"
            Reference observational layout.  Supported values include
            ``"Chabanier2019"`` and DESI Y1 labels.
        z_min, z_max : float, default: 0, 10
            Inclusive redshift interval retained in the forecast.
        add_noise : bool, default: False
            Add one Gaussian realization drawn from the reference covariance.
        seed : int, default: 0
            Random seed used when ``add_noise`` is true.

        Raises
        ------
        ValueError
            If ``data_label`` is unsupported.
        """

        # load covariance from data file
        self.data_label = data_label
        if data_label == "Chabanier2019":
            data_from_obs = data_Chabanier2019.read_from_file()
        # elif data_label == "Karacayli2024":
        #     data_from_obs = data_Karacayli2024.read_from_file()
        elif "DESIY1" in data_label:
            data_from_obs = data_DESIY1.P1D_DESIY1(data_label=data_label)
        else:
            raise ValueError("Unknown data_label", data_label)

        # evaluate theory at k_kms, for all redshifts. get Pk_kms from emulator
        zs = np.array(data_from_obs.z)
        theory.model_igm.set_fid_igm(zs)
        theory.set_fid_cosmo(zs)
        Pk_kms = theory.get_p1d_kms(zs, data_from_obs.k_kms, return_blob=False)
        full_Pk_kms = np.concatenate(np.array(Pk_kms, dtype=object)).reshape(-1)

        super().__init__(
            zs=data_from_obs.z,
            k_kms=data_from_obs.k_kms,
            Pk_kms=Pk_kms,
            cov_Pk_kms=data_from_obs.cov_Pk_kms,
            cov_stat=data_from_obs.covstat_Pk_kms,
            add_noise=add_noise,
            seed=seed,
            z_min=z_min,
            z_max=z_max,
            full_zs=data_from_obs.full_zs,
            full_Pk_kms=full_Pk_kms,
            full_cov_kms=data_from_obs.full_cov_Pk_kms,
            full_cov_stat_kms=data_from_obs.full_cov_stat_Pk_kms,
            theory=theory,
        )
