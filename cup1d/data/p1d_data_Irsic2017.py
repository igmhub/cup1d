import os
import numpy as np
from cup1d.data import p1d_data_base

class P1D_Irsic2017(p1d_data_base.BaseDataP1D):
    """Class containing P1D from Irsic et al. (2017)."""

    def __init__(self,add_syst=True,ignore_zcov=True):
        """Read measured P1D from Irsic et al. (2017).
         - add_syst=True will include systematic errors to covariance.
         - ignore_zcov=False will include covariance between z bins."""

        # folder storing P1D measurement
        assert ('CUP1D_PATH' in os.environ),'You need to define CUP1D_PATH'
        basedir=os.environ['CUP1D_PATH']+'/data_files/Irsic2017/'

        z,k_kms,Pk_kms,cov_Pk_kms=self._setup_from_file(basedir,add_syst,
                ignore_zcov)

        p1d_data_base.BaseDataP1D.__init__(self,z,k_kms,Pk_kms,cov_Pk_kms)

        return


    def _setup_from_file(self,basedir,add_syst,ignore_zcov):
        """Reconstruct measurement and covariance matrix from files."""
    
        assert ignore_zcov, 'implement cross-z covariance in p1d_Irsic2017'

        p1d_file=basedir+'/pk_xs_final.txt'
        inz,ink,inPk,inPkstat,inPksyst,_,_=np.loadtxt(p1d_file,unpack=True)
        # store unique values of redshift and wavenumber
        z=np.unique(inz)
        Nz=len(z)
        k_kms=np.unique(ink)
        Nk=len(k_kms)

        # store P1D, statistical error, noise power, metal power and systematic 
        Pk_kms=np.reshape(inPk,[Nz,Nk])
        Pkstat=np.reshape(inPkstat,[Nz,Nk])
        Pksyst=np.reshape(inPksyst,[Nz,Nk])

        # read covariance with statistical uncertainty
        cov_file=basedir+'/cov_pk_xs_final.txt'
        _,_,inCov=np.loadtxt(cov_file,unpack=True)
        cov_syst=inCov.reshape(Nz*Nk,Nz*Nk)

        # for now use diagonal covariance matrices
        cov_Pk_kms=[]
        for iz in range(Nz):
            # get covariance for z bin only
            zcov = cov_syst[iz*Nk:(iz+1)*Nk,iz*Nk:(iz+1)*Nk]
            if add_syst:
                zcov += np.diag(Pksyst[iz]**2)
            cov_Pk_kms.append(zcov)
                
        return z,k_kms,Pk_kms,cov_Pk_kms


