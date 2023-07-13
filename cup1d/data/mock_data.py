from lace.emulator import gp_emulator
from cup1d.data import base_p1d_data
from cup1d.data import data_Chabanier2019
from cup1d.data import data_Karacayli2022
from cup1d.data import data_QMLE_Ohio
from cup1d.likelihood import lya_theory


class Mock_P1D(base_p1d_data.BaseDataP1D):
    """ Class to generate a mock P1D from another P1D object and a theory"""

    def __init__(self,emulator=None,data_label="Chabanier2019",
                    zmin=2.0,zmax=4.5):
        """ Copy data and replace P1D signal using theory"""

        # load original data
        self.data_label=data_label
        if data_label=="Chabanier2019":
            data=data_Chabanier2019.P1D_Chabanier2019(zmin=zmin,zmax=zmax)
        elif data_label=="QMLE_Ohio":
            data=data_QMLE_Ohio.P1D_QMLE_Ohio(zmin=zmin,zmax=zmax)
        elif data_label=="Karacayli2022":
            data=data_Karacayli2022.P1D_Karacayli2022(zmin=zmin,zmax=zmax)
        else:
            raise ValueError("Unknown data_label",data_label)

        # check if emulator was provided
        if emulator is None:
            emulator=gp_emulator.GPEmulator()

        # setup and store theory (we will need it later)
        self.theory=lya_theory.Theory(zs=data.z,emulator=emulator)

        # at each z will update value of p1d
        Pk_kms=data.Pk_kms.copy()

        # evaluate theory at k_kms, for all redshifts
        emu_p1d_kms=self.theory.get_p1d_kms(data.k_kms)
        for iz,z in enumerate(data.z):
            Pk_kms[iz]=emu_p1d_kms[iz]

        base_p1d_data.BaseDataP1D.__init__(self,z=data.z,k_kms=data.k_kms,
                                    Pk_kms=Pk_kms,cov_Pk_kms=data.cov_Pk_kms)

