import os
import numpy as np
import pandas

from cup1d.data.base_p1d_data import BaseDataP1D, _drop_zbins


class P1D_QMLE_Ohio(BaseDataP1D):

    def __init__(
            self, diag_cov=True, kmin_kms=0.001, kmax_kms=0.04,
            zmin=None,zmax=None,version='ohio-v0', filename=None
    ):
        """Read measured P1D from file from Ohio mocks (QMLE)
        
        Args:
            filename: if not None, read that file.
        """

        # read redshifts, wavenumbers, power spectra and covariance matrices
        z,k,Pk,cov=self._read_file(diag_cov, kmin_kms, kmax_kms, version, filename)

        # drop low-z or high-z bins
        if zmin or zmax:
            z,k,Pk,cov=_drop_zbins(z,k,Pk,cov,zmin,zmax)

        super().__init__(z,k,Pk,cov)

        return


    def _read_file(self, diag_cov, kmin_kms, kmax_kms, version, filename):
        """Read file containing mock P1D"""

        if filename:
            fname = filename
        else:
            # DESI members can access this data in GitHub (cosmodesi/p1d_forecast)
            assert ('P1D_FORECAST' in os.environ),'Define P1D_FORECAST variable'
            basedir=os.environ['P1D_FORECAST']+'/private_data/p1d_measurements/'
            datadir=basedir+'/QMLE_Ohio/'

            # for now we can only handle diagonal covariances
            if version=='ohio-v0':
                fname=datadir+'/desi-y5fp-1.5-4-o3-deconv-power-qmle_kmax0.04.txt'
            else:
                raise ValueError('unknown version of DESI P1D '+version)
    
        # start by reading the file with measured band power
        print('will read P1D file', fname)
        assert os.path.isfile(fname), 'Ask Naim for P1D file'
        
        data = pandas.read_table(
            fname, comment='#', delim_whitespace=True
        ).to_records(index=False)
        # z k1 k2 kc Pfid ThetaP Pest ErrorP d b t
        zbins = np.unique(data['z'])
        kbins = np.unique(data['kc'])
        Nk = kbins.size
        Nz = zbins.size
        Pk = data['Pest'].reshape(Nz, Nk)
        
        assert (Nk * Nz == data.size)

        # will keep only wavenumbers with kmin_kms <= k <= kmax_kms
        drop_lowk = kbins < kmin_kms
        Nlk = np.sum(drop_lowk)
        if Nlk > 0:
            print(Nlk, 'low-k bins not included')

        drop_highk = kbins > kmax_kms
        Nhk = np.sum(drop_highk)
        if Nhk > 0:
            print(Nhk, 'high-k bins not included')

        kbins = kbins[Nlk:Nk - Nhk]
        Pk = Pk[:, Nlk:Nk - Nhk]

        # now read covariance matrix
        assert diag_cov, 'implement code to read full covariance'

        # for now only use diagonal elements
        cov = []
        for i in range(Nz):
            err = data['ErrorP'][i * Nk:(i + 1) * Nk]
            var = err[Nlk:Nk - Nhk]**2
            cov.append(np.diag(var))

        return zbins, kbins, Pk, cov
