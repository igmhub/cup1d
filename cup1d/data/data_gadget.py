import numpy as np
from scipy.interpolate import interp1d

from lace.emulator import poly_p1d
from lace.cosmo import camb_cosmo

from cup1d.data import base_p1d_data
from cup1d.data import data_PD2013
from cup1d.data import data_Chabanier2019
from cup1d.data import data_QMLE_Ohio
from cup1d.data import data_Karacayli2022

class Gadget_P1D(base_p1d_data.BaseDataP1D):
    """ Class to load an MP-Gadget simulation as a mock data object. 
        Can use PD2013 or Chabanier2019 covmats """

    def __init__(self,archive,
            sim_label="mpg_central",
            z_max=None,
            data_cov_label="Chabanier2019",
            data_cov_factor=1.0,
            add_syst=True,
            polyfit_kmax_Mpc=4.0,
            polyfit_ndeg=5):
        """ Read mock P1D from MP-Gadget sims, and returns mock measurement:
            - archive: p1d measurements from Gadget sims
            - sim_label: check available options in archive
            - z_max: maximum redshift to use in mock data
            - data_cov_label: P1D covariance to use (Chabanier2019 or PD2013)
            - data_cov_factor: multiply covariance by this factor
            - add_syst: Include systematic estimates in covariance matrices
            - polyfit_kmax_Mpc: kmax to use in polyfit (None for no polyfit)
            - polyfit_ndeg: poly degree to use in polyfit (None for no polyfit)
        """

        # covariance matrix settings
        self.add_syst=add_syst
        self.data_cov_factor=data_cov_factor
        self.data_cov_label=data_cov_label

        # polyfit settings
        self.polyfit_kmax_Mpc=polyfit_kmax_Mpc
        self.polyfit_ndeg=polyfit_ndeg

        # read P1D from simulation
        self.archive=archive
        self.testing_data = archive.get_testing_data(sim_label,z_max=z_max)

        # store cosmology used in the simulation (needs to be implemented)
        if sim_label in ["mpg_central", "mpg_seed", "mpg_reio"]:
            # use default cosmology in central simulation
            self.sim_cosmo=camb_cosmo.get_cosmology(ns=0.97,As=2e-9)
        elif sim_label == "mpg_growth":
            self.sim_cosmo=camb_cosmo.get_cosmology(ns=0.97,As=2e-9,H0=74)
        else:
            raise ValueError("need to use actual cosmology from sim")

        # setup P1D using covariance and testing sim
        z,k,Pk,cov=self._load_p1d(sim_label)

        # setup base class
        base_p1d_data.BaseDataP1D.__init__(self,z,k,Pk,cov)


    def _load_p1d(self,sim_label):

        # figure out dataset to mimic
        if self.data_cov_label=="Chabanier2019":
            data=data_Chabanier2019.P1D_Chabanier2019(add_syst=self.add_syst)
        elif self.data_cov_label=="PD2013":
            data=data_PD2013.P1D_PD2013(add_syst=self.add_syst)
        elif self.data_cov_label=="QMLE_Ohio":
            data=data_QMLE_Ohio.P1D_QMLE_Ohio()
        elif self.data_cov_label=="Karacayli2022":
            data=data_Karacayli2022.P1D_Karacayli2022()
        else:
            raise ValueError("Unknown data_cov_label",self.data_cov_label)

        k_kms=data.k_kms
        z_data=data.z

        # get redshifts in testing simulation
        z_sim=np.array([data['z'] for data in self.testing_data])
        zmin_sim=min(z_sim)

        # use simulation cosmology to convert units
        sim_camb_results=camb_cosmo.get_camb_results(self.sim_cosmo)

        # unit conversion, at zmin to get lowest possible k_min_kms
        dkms_dMpc_zmin=sim_camb_results.hubble_parameter(zmin_sim)/(1+zmin_sim)

        # Get k_min for the sim data & cut k values below that
        k_min_Mpc=self.testing_data[0]['k_Mpc'][0]
        if k_min_Mpc==0:
            k_min_Mpc=self.testing_data[0]['k_Mpc'][1]
        k_min_kms=k_min_Mpc/dkms_dMpc_zmin
        Ncull=np.sum(k_kms<k_min_kms)
        k_kms=k_kms[Ncull:]

        Pk_kms=[]
        cov=[]
        zs=[]
        # Set P1D and covariance for each redshift (from low-z to high-z)
        for iiz in range(len(z_sim)):
            # this is needed because sims have high-z first
            iz=len(z_sim)-1-iiz
            z=z_sim[iz]
            # convert Mpc to km/s
            dkms_dMpc=sim_camb_results.hubble_parameter(z)/(1+z)
            data_k_Mpc=k_kms*dkms_dMpc

            # find testing data for this redshift
            test_data=self.testing_data[iz]
            sim_k_Mpc=test_data['k_Mpc']
            sim_p1d_Mpc=test_data['p1d_Mpc']

            # mask k=0 if present
            if sim_k_Mpc[0]==0:
                sim_k_Mpc=sim_k_Mpc[1:]
                sim_p1d_Mpc=sim_p1d_Mpc[1:]

            # use polyfit instead of actual P1D from sim (unless asked not to)
            if self.polyfit_ndeg==None or self.polyfit_kmax_Mpc==None:
                # evaluate P1D in data wavenumbers (in velocity units)
                interp_sim_Mpc=interp1d(sim_k_Mpc,sim_p1d_Mpc,"cubic")
                sim_p1d_kms=interp_sim_Mpc(data_k_Mpc)*dkms_dMpc
            else:
                fit_p1d=poly_p1d.PolyP1D(sim_k_Mpc,sim_p1d_Mpc,
                            kmin_Mpc=1e-10,kmax_Mpc=self.polyfit_kmax_Mpc,
                            deg=self.polyfit_ndeg)
                # evalute polyfit to data wavenumbers
                sim_p1d_kms=fit_p1d.P_Mpc(data_k_Mpc)*dkms_dMpc

            # append redshift, p1d and covar
            zs.append(z)
            Pk_kms.append(sim_p1d_kms)

            # Now get covariance from the nearest z bin in data
            cov_mat=data.get_cov_iz(np.argmin(abs(z_data-z)))
            # Cull low k cov data and multiply by input factor
            cov_mat=self.data_cov_factor*cov_mat[Ncull:,Ncull:]
            cov.append(cov_mat)

        return zs, k_kms, Pk_kms, cov