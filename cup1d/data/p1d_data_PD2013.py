import os
import numpy as np
from cup1d.data import p1d_data_base

class P1D_PD2013(p1d_data_base.BaseDataP1D):
    """Class containing P1D from Palanque-Delabrouille et al. (2013)."""

    def __init__(self,use_FFT=True,add_syst=True):
        """Read measured P1D from Palanque-Delabrouille et al. (2013).
            If use_FFT=False, use likelihood version.
        """

        # folder storing P1D measurement
        assert ('CUP1D_PATH' in os.environ),'You need to define CUP1D_PATH'
        basedir=os.environ['CUP1D_PATH']+'/data_files/PD2013/'

        if use_FFT:
            z,k_kms,Pk_kms,cov_Pk_kms=self._setup_FFT(basedir,add_syst)
        else:
            z,k_kms,Pk_kms,cov_Pk_kms=self._setup_like(basedir,add_syst)

        p1d_data_base.BaseDataP1D.__init__(self,z,k_kms,Pk_kms,cov_Pk_kms)

        return


    def _setup_FFT(self,basedir,add_syst):
        """Setup measurement using FFT approach."""
    
        # start by reading Pk file
        p1d_file=basedir+'/table4a.dat'
        iz,ik,inz,ink,inPk,inPkstat,inPknoise,inPkmetal,inPksyst=np.loadtxt(
                    p1d_file,unpack=True)

        # store unique values of redshift and wavenumber
        z=np.unique(inz)
        Nz=len(z)
        k_kms=np.unique(ink)
        Nk=len(k_kms)

        # store P1D, statistical error, noise power, metal power and systematic 
        Pk_kms=np.reshape(inPk,[Nz,Nk])
        Pkstat=np.reshape(inPkstat,[Nz,Nk])    
        Pknoise=np.reshape(inPknoise,[Nz,Nk])
        Pkmetal=np.reshape(inPkmetal,[Nz,Nk])
        Pksyst=np.reshape(inPksyst,[Nz,Nk])

        # now read correlation matrices and compute covariance matrices
        cov_Pk_kms=[]
        for i in range(Nz):
            corr_file=basedir+'/cct4b'+str(i+1)+'.dat'
            corr=np.loadtxt(corr_file,unpack=True)
            # compute covariance matrix (stats only)
            sigma=Pkstat[i]
            zcov=np.multiply(sigma,np.multiply(corr,sigma))
            if add_syst:
                syst=Pksyst[i]
                zcov+=np.diag(syst)
            cov_Pk_kms.append(zcov)

        return z,k_kms,Pk_kms,cov_Pk_kms
        

    def _setup_like(self,basedir,add_syst):
        """Setup measurement using likelihood approach"""

        p1d_file=basedir+'/table5a.dat'
        raise ValueError('implement _setup_like to read likelihood P1D') 

