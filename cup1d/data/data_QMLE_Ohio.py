import os
import numpy as np
from cup1d.data import base_p1d_data

class P1D_QMLE_Ohio(base_p1d_data.BaseDataP1D):

    def __init__(self,diag_cov=True,kmax_kms=0.04,version='ohio-v0'):
        """Read measured P1D from file from Ohio mocks (QMLE)"""

        # read redshifts, wavenumbers, power spectra and covariance matrices
        z,k,Pk,cov=self._read_file(diag_cov,kmax_kms,version)

        base_p1d_data.BaseDataP1D.__init__(self,z,k,Pk,cov)

        return


    def _read_file(self,diag_cov,kmax_kms,version):
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

        # store measured P1D
        inPk=[float(line.split()[6]) for line in data]
        Pk=np.array(inPk).reshape([Nz,Nk])

        # will keep only wavenumbers with k < kmax_kms
        kmask=k<kmax_kms
        k=k[kmask]
        Nkmask=len(k)
        print('will only use {} k bins below {}'.format(Nkmask,kmax_kms))
        Pk=Pk[:,:Nkmask]

        # now read covariance matrix
        assert diag_cov, 'implement code to read full covariance'

        # for now only use diagonal elements
        inErr=[float(line.split()[7]) for line in data]
        cov=[]
        for i in range(Nz):
            err=inErr[i*Nk:(i+1)*Nk]
            var=np.array(err)[:Nkmask]**2
            cov.append(np.diag(var))

        return z,k,Pk,cov
