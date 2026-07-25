"""Interface between other emulators and the input emulator to cup1d likelihood

The LaCE emulators are compatible by default, this module is for other emulators
"""

import numpy as np

from lace.cosmo import cosmology
from forestflow.P3D_cINN import P3DEmulator
from forestflow.model_p3d_arinyo import ArinyoModel


class P1D_emulator:
    def __init__(self, name_emu="forest_mpg"):

        self.emulator = P3DEmulator(key=name_emu)

        self.kp_Mpc = 0.7
        self.list_sim_cube = []
        for ii in range(30):
            self.list_sim_cube.append(f"mpg_{ii}")
        self.zmax = 4.1  # data at z>4 is noisy
        self.kmax_Mpc = 4.0
        self.emu_params = self.emulator.input_labels
        self.emulator_label = name_emu

        self.arr_z = None
        self.arr_k_Mpc = None
        self.cosmo_params_dict = None
        self.model_Arinyo = None
        self.linear = None

    def set_cosmo(self, cosmo_params_dict):
        self.cosmo_params_dict = cosmo_params_dict
        fid_cosmo = cosmology.Cosmology(cosmo_params_dict=cosmo_params_dict)
        self.model_Arinyo = ArinyoModel(fid_cosmo)

    def set_linear_theory(self, z, new_cosmo_params=None):

        z = np.atleast_1d(z)

        if (self.linear is not None) and same_cosmo(
            self.cosmo_params_dict, new_cosmo_params
        ):
            return

        self.linear = self.model_Arinyo.linear_theory(
            zmin=z.min(),
            zmax=z.max(),
            new_cosmo_params=new_cosmo_params,
        )

    def emulate_p1d_Mpc(self, zs, kin_Mpc, in_params):

        list_dicts = []
        nin = in_params["Delta2_p"].shape[0]
        for ii in range(nin):
            in_par_only = {}
            for par in self.emu_params:
                in_par_only[par] = in_params[par][ii]
            list_dicts.append(in_par_only)
        out_emu = self.emulator.evaluate(list_dicts)

        list_P1D_Mpc = self.model_Arinyo.P1D_Mpc(
            self.linear,
            zs,
            kin_Mpc,
            out_emu,
        )

        return list_P1D_Mpc


def same_cosmo(cosmo_params_dict, new_cosmo_params):
    if new_cosmo_params is None:
        return True

    return all(
        cosmo_params_dict.get(key) == value for key, value in new_cosmo_params.items()
    )
