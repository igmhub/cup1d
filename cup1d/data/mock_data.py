from cup1d.data import base_p1d_data

class Mock_P1D(base_p1d_data.BaseDataP1D):
    """ Class to generate a mock P1D from another P1D object and a theory"""

    def __init__(self,data,theory):
        """ Copy data and replace P1D signal using theory"""

        # evaluate theory at k_kms, for all redshifts
        emu_p1d_kms=theory.get_p1d_kms(data.k_kms)

        # at each z, update value of p1d
        Pk_kms=data.Pk_kms.copy()
        for iz,z in enumerate(data.z):
            Pk_kms[iz]=emu_p1d_kms[iz]

        # copy data
        base_p1d_data.BaseDataP1D.__init__(self,z=data.z,k_kms=data.k_kms,
                                    Pk_kms=Pk_kms,cov_Pk_kms=data.cov_Pk_kms)

