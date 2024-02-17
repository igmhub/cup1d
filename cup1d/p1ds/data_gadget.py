import numpy as np
from scipy.interpolate import interp1d

from lace.utils import poly_p1d
from lace.cosmo import camb_cosmo
from cup1d.p1ds.base_p1d_mock import BaseMockP1D
from cup1d.p1ds import (
    data_PD2013,
    data_Chabanier2019,
    data_QMLE_Ohio,
    data_Karacayli2022,
)


class Gadget_P1D(BaseMockP1D):
    """Class to load an MP-Gadget simulation as a mock data object.
    Can use PD2013 or Chabanier2019 covmats"""

    def __init__(
        self,
        testing_data,
        emulator=None,
        apply_smoothing=True,
        input_sim="mpg_central",
        z_min=None,
        z_max=None,
        data_cov_label="Chabanier2019",
        data_cov_factor=1.0,
        add_syst=True,
        polyfit_kmax_Mpc=4.0,
        polyfit_ndeg=5,
        add_noise=False,
        seed=0,
    ):
        """Read mock P1D from MP-Gadget sims, and returns mock measurement:
        - testing_data: p1d measurements from Gadget sims
        - input_sim: check available options in testing_data
        - z_max: maximum redshift to use in mock data
        - data_cov_label: P1D covariance to use (Chabanier2019 or PD2013)
        - data_cov_factor: multiply covariance by this factor
        - add_syst: Include systematic estimates in covariance matrices
        - polyfit_kmax_Mpc: kmax to use in polyfit (None for no polyfit)
        - polyfit_ndeg: poly degree to use in polyfit (None for no polyfit)
        """

        # covariance matrix settings
        self.add_syst = add_syst
        self.data_cov_factor = data_cov_factor
        self.data_cov_label = data_cov_label

        # store sim data
        self.input_sim = input_sim

        if apply_smoothing & (emulator is not None):
            self.testing_data = super().set_smoothing_Mpc(
                emulator, testing_data
            )
        else:
            self.testing_data = testing_data

        # store cosmology used in the simulation
        dkms_dMpc = []
        for ii in range(len(testing_data)):
            dkms_dMpc.append(testing_data[ii]["dkms_dMpc"])
        self.dkms_dMpc = np.array(dkms_dMpc)

        cosmo_params = self.testing_data[0]["cosmo_params"]
        self.sim_cosmo = camb_cosmo.get_cosmology_from_dictionary(cosmo_params)

        # setup P1D using covariance and testing sim
        z, k, Pk, cov = self._load_p1d()

        # setup base class
        super().__init__(z, k, Pk, cov, add_noise=add_noise, seed=seed)

        return

    def _load_p1d(self):
        # figure out dataset to mimic
        if self.data_cov_label == "Chabanier2019":
            data = data_Chabanier2019.P1D_Chabanier2019(add_syst=self.add_syst)
        elif self.data_cov_label == "PD2013":
            data = data_PD2013.P1D_PD2013(add_syst=self.add_syst)
        elif self.data_cov_label == "QMLE_Ohio":
            data = data_QMLE_Ohio.P1D_QMLE_Ohio()
        elif self.data_cov_label == "Karacayli2022":
            data = data_Karacayli2022.P1D_Karacayli2022()
        else:
            raise ValueError("Unknown data_cov_label", self.data_cov_label)

        k_kms = data.k_kms
        z_data = data.z

        # get redshifts in testing simulation
        z_sim = np.array([data["z"] for data in self.testing_data])

        # unit conversion, at zmin to get lowest possible k_min_kms
        dkms_dMpc_zmin = self.dkms_dMpc[np.argmin(z_sim)]

        # Get k_min for the sim data & cut k values below that
        k_min_Mpc = self.testing_data[0]["k_Mpc"][0]
        if k_min_Mpc == 0:
            k_min_Mpc = self.testing_data[0]["k_Mpc"][1]
        k_min_kms = k_min_Mpc / dkms_dMpc_zmin
        Ncull = np.sum(k_kms < k_min_kms)
        k_kms = k_kms[Ncull:]

        Pk_kms = []
        cov = []
        zs = []
        for iz in range(len(z_sim)):
            z = z_sim[iz]

            # convert Mpc to km/s
            data_k_Mpc = k_kms * self.dkms_dMpc[iz]

            # find testing data for this redshift
            sim_k_Mpc = self.testing_data[iz]["k_Mpc"].copy()
            sim_p1d_Mpc = self.testing_data[iz]["p1d_Mpc"].copy()

            # mask k=0 if present
            if sim_k_Mpc[0] == 0:
                sim_k_Mpc = sim_k_Mpc[1:]
                sim_p1d_Mpc = sim_p1d_Mpc[1:]

            interp_sim_Mpc = interp1d(sim_k_Mpc, sim_p1d_Mpc, "cubic")
            sim_p1d_kms = interp_sim_Mpc(data_k_Mpc) * self.dkms_dMpc[iz]

            # append redshift, p1d and covar
            zs.append(z)
            Pk_kms.append(sim_p1d_kms)

            # Now get covariance from the nearest z bin in data
            cov_mat = data.get_cov_iz(np.argmin(abs(z_data - z)))
            # Cull low k cov data and multiply by input factor
            cov_mat = self.data_cov_factor * cov_mat[Ncull:, Ncull:]
            cov.append(cov_mat)

        return zs, k_kms, Pk_kms, cov
