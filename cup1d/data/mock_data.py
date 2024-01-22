from lace.emulator import gp_emulator
from cup1d.data.base_p1d_mock import BaseMockP1D
from cup1d.data import data_Chabanier2019
from cup1d.data import data_Karacayli2022
from cup1d.data import data_QMLE_Ohio
from cup1d.likelihood import lya_theory


class Mock_P1D(BaseMockP1D):
    """Class to generate a mock P1D from another P1D object and a theory"""

    def __init__(
        self,
        emulator=None,
        data_label="Chabanier2019",
        zmin=2.0,
        zmax=4.5,
        add_noise=False,
        seed=0,
        fid_sim_igm="mpg_central",
        **kwargs
    ):
        """Copy data and replace P1D signal using theory
        Args:
            **kwargs: Any other key words arguments to pass into data constructor.
        """

        # load original data
        self.data_label = data_label
        if data_label == "Chabanier2019":
            data = data_Chabanier2019.P1D_Chabanier2019(zmin=zmin, zmax=zmax)
        elif data_label == "QMLE_Ohio":
            data = data_QMLE_Ohio.P1D_QMLE_Ohio(zmin=zmin, zmax=zmax, **kwargs)
        elif data_label == "Karacayli2022":
            data = data_Karacayli2022.P1D_Karacayli2022(zmin=zmin, zmax=zmax)
        else:
            raise ValueError("Unknown data_label", data_label)

        # check if emulator was provided
        if emulator is None:
            emulator = gp_emulator.GPEmulator(training_set="Pedersen21")
            print("Using default emulator: Pedersen21")

        # setup and store theory (we will need it later)
        self.theory = lya_theory.Theory(
            zs=data.z, emulator=emulator, fid_sim_igm=fid_sim_igm
        )

        # at each z will update value of p1d
        Pk_kms = data.Pk_kms.copy()

        # evaluate theory at k_kms, for all redshifts
        emu_p1d_kms = self.theory.get_p1d_kms(data.k_kms)
        for iz, z in enumerate(data.z):
            Pk_kms[iz] = emu_p1d_kms[iz]

        super().__init__(
            z=data.z,
            k_kms=data.k_kms,
            Pk_kms=Pk_kms,
            cov_Pk_kms=data.cov_Pk_kms,
            add_noise=False,
            seed=0,
        )
