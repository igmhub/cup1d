import os
import numpy as np
from cup1d.data import p1d_data_base

class P1D_Walther2018(p1d_data_base.BaseDataP1D):
    """Class containing P1D from Walther et al. (2018)."""

    def __init__(self):
        """Read measured P1D from Walther et al. (2018)."""

        # folder storing P1D measurement
        assert ('CUP1D_PATH' in os.environ),'You need to define CUP1D_PATH'
        basedir=os.environ['CUP1D_PATH']+'/data_files/Walther2018/'

        z,k_kms,Pk_kms,cov_Pk_kms=self._setup_from_file(basedir)

        p1d_data_base.BaseDataP1D.__init__(self,z,k_kms,Pk_kms,cov_Pk_kms)

        return


    def _setup_from_file(self,basedir):
        """Reconstruct covariance matrix from files."""
    
        # start by reading Pk file
        p1d_file=basedir+'/table5.dat'

        # note that the file contains    k P1D(k) / pi
        inz,ink,inkPk,inkPkstat=np.loadtxt(p1d_file,unpack=True)
 
        # divide by wavenumber and multiply by pi to get flux power (and error)
        inPk=inkPk/ink*np.pi
        inPkstat=inkPkstat/ink*np.pi

        # store unique values of redshift and wavenumber
        z=np.unique(inz)
        Nz=len(z)

        # wavenumbers vary slightly between redshifts, compute mean
        Nk=int(len(inz)/Nz)
        k_kms=np.mean(ink.reshape(Nz,Nk),axis=0)

        # store P1D and statistical error
        Pk_kms=np.reshape(inPk,[Nz,Nk])
        Pkstat=np.reshape(inPkstat,[Nz,Nk])

        # now read correlation matrices
        corr_file=basedir+'/table7.dat'
        incorr=np.loadtxt(corr_file,unpack=True)

        # compute covariance matrices, one for redshift bin 
        cov_Pk_kms=[]
        for iz in range(Nz):
            # get correlation matrix for this redshift bin
            corr=incorr[2:,iz*Nk:(iz+1)*Nk]
            # compute covariance matrix (stats only)
            sigma=Pkstat[iz]
            zcov=np.multiply(sigma,np.multiply(corr,sigma))
            cov_Pk_kms.append(zcov)

        return z,k_kms,Pk_kms,cov_Pk_kms
        

