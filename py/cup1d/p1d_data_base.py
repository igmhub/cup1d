import numpy as np
import matplotlib.pyplot as plt


class BaseDataP1D(object):
    """Base class to store measurements of the 1D power spectrum."""


    def __init__(self,z,k_kms,Pk_kms,cov_Pk_kms):
        """Construct base P1D class, from measured power and covariance"""

        self.z=z
        self.k_kms=k_kms
        self.Pk_kms=Pk_kms
        self.cov_Pk_kms=cov_Pk_kms


    def get_Pk_iz(self,iz):
        """Return P1D in units of km/s for redshift bin iz."""

        return self.Pk_kms[iz]


    def get_cov_iz(self,iz):
        """Return covariance of P1D in units of (km/s)^2 for redshift bin iz."""

        return self.cov_Pk_kms[iz]


    def _cull_data(self,min_kp_kms):
        """Remove bins with wavenumber k < min_kp_kms."""

        # figure out number of bins to cull
        Ncull=np.sum(self.k_kms<min_kp_kms)

        # cull wavenumbers, power spectra, and covariances
        self.k_kms=self.k_kms[Ncull:]
        self.Pk_kms=self.Pk_kms[:,Ncull:]
        for i in range(len(self.z)):
            self.cov_Pk_kms[i]=self.cov_Pk_kms[i][Ncull:,Ncull:]

        return


    def plot_p1d(self,use_dimensionless=True):
        """Plot P1D mesurement. If use_dimensionless, plot k*P(k)/pi."""

        N=len(self.z)
        for i in range(N):
            k_kms=self.k_kms
            Pk_kms=self.get_Pk_iz(i)
            err_Pk_kms=np.sqrt(np.diagonal(self.get_cov_iz(i)))
            if use_dimensionless:
                fact=k_kms/np.pi
            else:
                fact=1.0
            plt.errorbar(k_kms,fact*Pk_kms,yerr=fact*err_Pk_kms,
                            label='z = {}'.format(self.z[i]))

        plt.legend()
        plt.yscale('log', nonposy='clip')
        plt.xlabel('k [s/km]')
        if use_dimensionless:
            plt.ylabel(r'$k P(k)/ \pi$')
        else:
            plt.ylabel('P(k) [km/s]')
        plt.show()

