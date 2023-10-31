import os
import numpy as np
from cup1d.data.base_p1d_data import BaseDataP1D, _drop_zbins


class P1D_QMLE_Ohio(BaseDataP1D):

    def __init__(self,diag_cov=True,kmin_kms=0.001,kmax_kms=0.04,
                zmin=None,zmax=None,version='ohio-v0'):
        """Read measured P1D from file from Ohio mocks (QMLE)"""

        # read redshifts, wavenumbers, power spectra and covariance matrices
        z,k,Pk,cov=self._read_file(diag_cov,kmin_kms,kmax_kms,version)

        # drop low-z or high-z bins
        if zmin or zmax:
            z,k,Pk,cov=_drop_zbins(z,k,Pk,cov,zmin,zmax)

        super().__init__(z,k,Pk,cov)

        return


    def _read_file(self,diag_cov,kmin_kms,kmax_kms,version):
        """Read file containing mock P1D"""

        # DESI members can access this data in GitHub (cosmodesi/p1d_forecast)
        assert ('P1D_FORECAST' in os.environ),'Define P1D_FORECAST variable'
        basedir=os.environ['P1D_FORECAST']+'/private_data/p1d_measurements/'
        datadir=basedir+'/QMLE_Ohio/'

        # for now we can only handle diagonal covariances
        if version=='ohio-v0':
            fname=datadir+'/desi-y5fp-1.5-4-o3-deconv-power-qmle_kmax0.04.txt'
            Nz=12
            Nk=32
            #first line in ascii file containing p1d
            istart=42
        else:
            raise ValueError('unknown version of DESI P1D '+version)
    
        # start by reading the file with measured band power
        print('will read P1D file',fname)
        assert os.path.isfile(fname), 'Ask Naim for P1D file'
        with open(fname, 'r') as reader:
            lines=reader.readlines()

        # z k1 k2 kc Pfid ThetaP Pest ErrorP d b t
        data = lines[istart:]

        # store unique redshifts 
        inz=[float(line.split()[0]) for line in data]
        z=np.unique(inz)
        # store unique wavenumbers 
        ink=[float(line.split()[3]) for line in data]
        k=np.unique(ink)
        Nk=len(k)

        # store measured P1D
        inPk=[float(line.split()[6]) for line in data]
        Pk=np.array(inPk).reshape([Nz,Nk])

        # will keep only wavenumbers with kmin_kms <= k <= kmax_kms
        drop_lowk=(k<kmin_kms)
        Nlk=np.sum(drop_lowk)
        if Nlk>0:
            print(Nlk,'low-k bins not included')
        drop_highk=(k>kmax_kms)
        Nhk=np.sum(drop_highk)
        if Nhk>0:
            print(Nhk,'high-k bins not included')
        k=k[Nlk:Nk-Nhk]
        Pk=Pk[:,Nlk:Nk-Nhk]

        # now read covariance matrix
        assert diag_cov, 'implement code to read full covariance'

        # for now only use diagonal elements
        inErr=[float(line.split()[7]) for line in data]
        cov=[]
        for i in range(Nz):
            err=inErr[i*Nk:(i+1)*Nk]
            var=np.array(err)[Nlk:Nk-Nhk]**2
            cov.append(np.diag(var))

        return z,k,Pk,cov
