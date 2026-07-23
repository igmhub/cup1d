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

    def set_cosmo(self, cosmo_params_dict):
        fid_cosmo = cosmology.Cosmology(cosmo_params_dict=cosmo_params_dict)
        self.model_Arinyo = ArinyoModel(fid_cosmo)

    def emulate_p1d_Mpc(self, in_params, k_Mpc, z, new_cosmo_params=None):

        arr_z = np.atleast_1d(z)
        arr_k_Mpc = np.atleast_1d(k_Mpc)

        list_dicts = []
        nin = in_params["Delta2_p"].shape[0]
        for ii in range(nin):
            in_par_only = {}
            for par in self.emu_params:
                in_par_only[par] = in_params[par][ii]
            list_dicts.append(in_par_only)

        out_emu = self.emulator.evaluate(list_dicts)

        list_out_emu = []
        for ii in range(nin):
            out_par_only = {}
            for par in out_emu:
                out_par_only[par] = out_emu[par][ii]
            list_out_emu.append(out_par_only)

        list_P1D_Mpc = []
        for ii in range(nin):
            _P1D_Mpc = self.model_Arinyo.P1D_Mpc(
                arr_z[ii],
                arr_k_Mpc[ii],
                list_out_emu[ii],
                new_cosmo_params=new_cosmo_params,
            )
            list_P1D_Mpc.append(_P1D_Mpc)

        return list_P1D_Mpc
