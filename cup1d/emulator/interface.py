"""Interface between other emulators and the input emulator to cup1d likelihood

The LaCE emulators are compatible by default, this module is for other emulators
"""

import numpy as np

from lace.cosmo import cosmology
from forestflow.P3D_cINN import P3DEmulator
from forestflow.model_p3d_arinyo import ArinyoModel


class P1D_emulator:
    def __init__(self, name_emu="forest_mpg", compile_model=True):

        self.emulator = P3DEmulator(key=name_emu, compile_model=compile_model)

        self.kp_iMpc = 0.7
        self.list_sim_cube = []
        for ii in range(30):
            self.list_sim_cube.append(f"mpg_{ii}")
        self.zmax = 4.1  # data at z>4 is noisy
        self.kmax_iMpc = 4.0
        self.emu_params = self.emulator.input_labels
        self.emulator_label = name_emu

        self.arr_z = None
        self.arr_k_iMpc = None
        self.cosmo_params_dict = None
        self.model_Arinyo = None
        self.linear = None
        self._prediction_cache = None

    def set_cosmo(self, cosmo_params_dict):
        self.cosmo_params_dict = cosmo_params_dict
        fid_cosmo = cosmology.Cosmology(cosmo_params_dict=cosmo_params_dict)
        self.model_Arinyo = ArinyoModel(fid_cosmo)

    def set_linear_theory(self, z, new_cosmo_params=None):

        z = np.atleast_1d(z)
        zuse = np.unique(z)

        if (self.linear is not None) and same_cosmo(
            self.cosmo_params_dict, new_cosmo_params
        ):
            if np.all(zuse == self.linear.z):
                return

        self.linear = self.model_Arinyo.linear_theory(
            zuse, new_cosmo_params=new_cosmo_params
        )

    def _prediction_key(self, parameters, latent_index=None):
        """Return a stable key for one set of emulator inputs."""

        values = tuple(float(parameters[name]) for name in self.emu_params)
        return (latent_index, values) if latent_index is not None else values

    def prime_prediction_cache(self, emulator_calls):
        """Evaluate many redshift inputs in one ForestFlow network batch."""

        unique_inputs = {}
        for emulator_call in emulator_calls:
            n_redshifts = np.asarray(emulator_call[self.emu_params[0]]).size
            for index in range(n_redshifts):
                parameters = {
                    name: np.asarray(emulator_call[name]).reshape(-1)[index]
                    for name in self.emu_params
                }
                key = self._prediction_key(parameters, latent_index=index)
                unique_inputs.setdefault(key, parameters)

        if not unique_inputs:
            self._prediction_cache = {}
            return

        self._prediction_cache = {}
        items = list(unique_inputs.items())
        for start in range(0, len(items), 128):
            chunk = items[start : start + 128]
            keys = [item[0] for item in chunk]
            predictions = self.emulator.evaluate(
                [item[1] for item in chunk],
                latent_indices=[key[0] for key in keys],
            )
            for index, key in enumerate(keys):
                self._prediction_cache[key] = {
                    name: np.asarray(predictions[name]).reshape(-1)[index]
                    for name in self.emulator.output_labels
                }

    def clear_prediction_cache(self):
        """Discard predictions retained for one batched likelihood call."""

        self._prediction_cache = None

    def emulate_p1d_Mpc(self, zs, kin_Mpc, in_params):

        list_dicts = []
        nin = in_params["Delta2_p"].shape[0]
        for ii in range(nin):
            in_par_only = {}
            for par in self.emu_params:
                in_par_only[par] = in_params[par][ii]
            list_dicts.append(in_par_only)
        if self._prediction_cache is None:
            out_emu = self.emulator.evaluate(list_dicts)
        else:
            cached = [
                self._prediction_cache[
                    self._prediction_key(parameters, latent_index=index)
                ]
                for index, parameters in enumerate(list_dicts)
            ]
            out_emu = {
                name: np.asarray([prediction[name] for prediction in cached])
                for name in self.emulator.output_labels
            }

        list_P1D_Mpc = self.model_Arinyo.P1D_Mpc(
            self.linear,
            zs,
            kin_Mpc,
            out_emu,
        )

        return list_P1D_Mpc

    def emulate_p1d_Mpc_batch(self, zs, kin_Mpc, in_params, cosmo_params_batch):
        """Evaluate ForestFlow for ``(batch, redshift, k)`` inputs.

        The cINN is evaluated once over flattened batch/redshift rows. Linear
        theory remains one inexpensive rescaling per cosmology because each
        Arinyo integration owns a distinct linear grid.
        """
        zs = np.atleast_1d(zs)
        n_batch, n_z, n_k = np.asarray(kin_Mpc).shape
        calls = [{name: np.asarray(in_params[name])[ib, iz] for name in self.emu_params}
                 for ib in range(n_batch) for iz in range(n_z)]
        output = self.emulator.evaluate(calls, latent_indices=np.tile(np.arange(n_z), n_batch))
        arinyo = {name: np.asarray(values).reshape(n_batch, n_z) for name, values in output.items()}
        linear = self.model_Arinyo.linear_theory_batch(zs, cosmo_params_batch)
        return self.model_Arinyo.P1D_Mpc(linear, zs, kin_Mpc, arinyo)

    def emulate_P1D_Mpc(self, zs, kin_iMpc, in_params):
        """Return P1D_Mpc for wavenumbers ``kin_iMpc`` in Mpc^-1."""
        return self.emulate_p1d_Mpc(zs, kin_iMpc, in_params)

    kp_Mpc = property(lambda self: self.kp_iMpc, lambda self, value: setattr(self, "kp_iMpc", value))
    kmax_Mpc = property(lambda self: self.kmax_iMpc, lambda self, value: setattr(self, "kmax_iMpc", value))
    arr_k_Mpc = property(lambda self: self.arr_k_iMpc, lambda self, value: setattr(self, "arr_k_iMpc", value))


def same_cosmo(cosmo_params_dict, new_cosmo_params):
    if new_cosmo_params is None:
        return True

    return all(
        cosmo_params_dict.get(key) == value for key, value in new_cosmo_params.items()
    )
