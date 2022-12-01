import os
import numpy as np
from cup1d.data import p1d_data_base

class P1D_Chabanier2019(p1d_data_base.BaseDataP1D):
    """Class containing P1D from Chabanier et al. (2019)."""

    def __init__(self,add_syst=True):
        """Read measured P1D from Chabanier et al. (2019)."""

        # folder storing P1D measurement
        assert ('CUP1D_PATH' in os.environ),'You need to define CUP1D_PATH'
        basedir=os.environ['CUP1D_PATH']+'/data_files/p1d_measurements/'
        datadir=basedir+'/Chabanier2019/'

        z,k_kms,Pk_kms,cov_Pk_kms=self._setup_from_file(datadir,add_syst)

        p1d_data_base.BaseDataP1D.__init__(self,z,k_kms,Pk_kms,cov_Pk_kms)

        return


    def _setup_from_file(self,basedir,add_syst):
        """Reconstruct covariance matrix from files."""
    
        # start by reading Pk file
        p1d_file=basedir+'/Pk1D_data.dat'
        inz,ink,inPk,inPkstat,inPknoise,inPkmetal=np.loadtxt(p1d_file,
                                                                unpack=True)

        # store unique values of redshift and wavenumber
        z=np.unique(inz)
        Nz=len(z)
        k_kms=np.unique(ink)
        Nk=len(k_kms)

        # continue by reading file with systematic uncertainties
        syst_file=basedir+'Pk1D_syst.dat'
        insyst=np.loadtxt(syst_file,unpack=True)
        # add in quadrature 8 different systematics
        Nsyst=insyst.shape[0]
        syst_var=np.zeros(Nz*Nk)
        for i in range(Nsyst):
            syst_var += (insyst[i,:]**2)

        # store P1D, statistical error, noise power, metal power and systematic 
        Pk_kms=np.reshape(inPk,[Nz,Nk])
        Pkstat=np.reshape(inPkstat,[Nz,Nk])    
        Pknoise=np.reshape(inPknoise,[Nz,Nk])
        Pkmetal=np.reshape(inPkmetal,[Nz,Nk])
        Pksyst=np.reshape(np.sqrt(syst_var),[Nz,Nk])

        # now read correlation matrices
        corr_file=basedir+'Pk1D_cor.dat'
        incorr=np.loadtxt(corr_file,unpack=True)
        # note strange order 
        allcorr=np.reshape(incorr,[Nk,Nz,Nk])

        # compute covariance matrices with statistics and systematic errors 
        cov_Pk_kms=[]
        for i in range(Nz):
            corr=allcorr[:,i,:]
            # compute covariance matrix (stats only)
            sigma=Pkstat[i]
            zcov=np.multiply(sigma,np.multiply(corr,sigma))
            if add_syst:
                syst=Pksyst[i]
                zcov+=np.diag(syst)
            cov_Pk_kms.append(zcov)

        return z,k_kms,Pk_kms,cov_Pk_kms
        

