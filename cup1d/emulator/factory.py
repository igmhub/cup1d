"""cup1d's single interface to the P1D emulators.

LaCE's Gaussian process predicts the P1D directly
(``emulate_p1d_Mpc(inputs, k)``); ForestFlow predicts Arinyo coefficients that
are projected to P1D with the linear power of a cosmology
(``emulate_p1d_Mpc(zs, k, inputs, cosmologies=...)``). :class:`Emulator`
gives both one batched interface, used by the native theory
(:class:`cup1d.theory.theory.Theory`) and by the Cobaya theory of
lya_interface, so that both evaluate the same emulator in the same way.
"""

import copy

import numpy as np

# cup1d labels of the LaCE central GP bundles
_LACE_ALIASES = {
    "lace_mpg": "CH24_mpgcen_gpr",
    "lace_nyx": "CH24_nyxcen_gpr",
}
FORESTFLOW_LABELS = ("forest_mpg_cinn", "forest_mpg_gp")
_REMOVED_LABELS = ("forest_mpg", "forest_mpg_fix", "forest_mpg_old")


class Emulator:
    """One P1D emulator (LaCE GP or ForestFlow) behind one batched interface.

    Parameters
    ----------
    label : str, default="lace_mpg"
        ``lace_mpg``/``lace_nyx`` (LaCE central GP bundles; other LaCE labels
        are passed to :func:`lace.emulator.emulator_manager.set_emulator`),
        ``forest_mpg_gp`` (ForestFlow Gaussian process) or
        ``forest_mpg_cinn`` (ForestFlow cINN).
    model : object, optional
        An already built LaCE or ForestFlow emulator (tests, custom bundles);
        ``label`` then only names it.
    **options
        Passed to ForestFlow's :class:`forestflow.emulator.P1DEmulator`
        (e.g. latent-sampling options of the cINN).

    Attributes
    ----------
    label : str
        The cup1d label.
    model : object
        The LaCE ``GPEmulator`` or ForestFlow ``P1DEmulator``. Attributes not
        defined here are read from it (e.g. ``input_norm`` of LaCE).
    forestflow : bool
        True for ForestFlow (Arinyo coefficients projected to P1D).
    """

    def __init__(self, label="lace_mpg", model=None, **options):
        if label in _REMOVED_LABELS:
            raise ValueError(
                f"emulator_label {label!r} was removed: use 'forest_mpg_cinn' "
                "(neural network) or 'forest_mpg_gp' (Gaussian process)"
            )
        self.label = label
        # ForestFlow: Arinyo coefficients projected to P1D (also for a given model)
        self.forestflow = label in FORESTFLOW_LABELS or (
            model is not None and hasattr(model, "evaluate_arinyo"))
        if model is not None:
            self.model = model
        elif self.forestflow:
            from forestflow.emulator import P1DEmulator

            self.model = P1DEmulator(name_emu=label, **options)
        else:
            if options:
                raise ValueError(f"options {sorted(options)} apply only to ForestFlow emulators")
            from lace.emulator import emulator_manager

            self.model = emulator_manager.set_emulator(
                emulator_label=_LACE_ALIASES.get(label, label)
            )

    def __getattr__(self, name):
        # only called for attributes not found on the interface itself
        if name == "model":
            raise AttributeError(name)
        return getattr(self.model, name)

    def __repr__(self):
        return f"Emulator({self.label!r})"

    # ------------------------------------------------------------------ inputs
    @property
    def emu_params(self):
        """Names of the emulator inputs."""
        return list(self.model.emu_params)

    @property
    def kp_Mpc(self):
        """Pivot of the linear-power inputs in 1/Mpc."""
        return self.model.kp_Mpc

    @property
    def emulator_label(self):
        """Label of the underlying bundle (names its covariance file)."""
        return self.model.emulator_label

    @property
    def suite(self):
        """Training simulation suite (``"mpg"`` or ``"nyx"``)."""
        return self.model.list_sim_cube[0][:3]

    @property
    def batches(self):
        """Whether batches of points are cheaper per point than single points.

        True for the LaCE GP (one call for all rows); ForestFlow evaluates
        one linear power per cosmology, so batching does not pay.
        """
        return not self.forestflow

    # ------------------------------------------------------------- predictions
    def set_cosmology(self, cosmology):
        """Fiducial cosmology of ForestFlow's linear power (no-op for LaCE)."""
        if self.forestflow:
            self.model.set_cosmology(cosmology)

    def independent_copy(self):
        """A copy whose cosmology can be set without changing this one.

        The trained model is shared (read only).
        """
        new = copy.copy(self)
        new.model = copy.copy(self.model)
        return new

    def arinyo(self, inputs):
        """Arinyo coefficients of ForestFlow for inputs of any shape.

        Parameters
        ----------
        inputs : mapping
            Emulator inputs, all of one shape (scalar, ``(n_z,)`` or
            ``(n_batch, n_z)``).

        Returns
        -------
        dict
            Coefficients with the shape of the inputs. A cINN draws the same
            latent block for every row, so native cup1d and the Cobaya theory
            get the same coefficients for the same inputs.

        Raises
        ------
        ValueError
            For an emulator that predicts the P1D directly (LaCE).
        """
        if not self.forestflow:
            raise ValueError(f"{self.label} predicts the P1D directly: no Arinyo coefficients")
        return self.model.evaluate_arinyo(inputs)

    def p1d_Mpc(self, zs, k_Mpc, inputs, cosmologies=None):
        """P1D in Mpc for a batch of points.

        Parameters
        ----------
        zs : array_like
            Redshifts.
        k_Mpc : ndarray, shape (n_batch, n_z, n_k)
            Parallel wavenumbers in 1/Mpc (zero padded rows allowed).
        inputs : mapping
            Emulator inputs of shape ``(n_batch, n_z)``.
        cosmologies : sequence, optional
            One LaCE cosmology per row; required by ForestFlow (its linear
            power), ignored by LaCE.

        Returns
        -------
        ndarray, shape (n_batch, n_z, n_k)
        """
        k_Mpc = np.asarray(k_Mpc, dtype=float)
        n_batch, n_z, n_k = k_Mpc.shape
        if self.forestflow:
            p1d = self.model.emulate_p1d_Mpc(zs, k_Mpc, inputs, cosmologies=cosmologies)
            return np.asarray(p1d).reshape(n_batch, n_z, n_k)
        names = getattr(self.model, "emu_params", None) or [name for name in inputs if name != "mF_fid"]
        flat = {name: np.asarray(inputs[name], dtype=float).reshape(-1) for name in names}
        p1d = self.model.emulate_p1d_Mpc(flat, k_Mpc.reshape(n_batch * n_z, n_k))
        return np.asarray(p1d).reshape(n_batch, n_z, n_k)

    def p1d_and_metal_moments_Mpc(self, zs, k_Mpc, inputs, cosmologies):
        """P1D and Kaiser metal moments in Mpc for a batch (ForestFlow only).

        One projection pass per point with ForestFlow's default quadrature
        (as :meth:`p1d_Mpc`); see
        :func:`cup1d.theory.projection.fused_projection_Mpc`.

        Returns
        -------
        p1d : ndarray, shape (n_batch, n_z, n_k)
        moments : dict of str to ndarray
            Each ``(n_batch, n_z, n_k)``.
        """
        if not self.forestflow:
            raise ValueError("the Kaiser metal model needs a ForestFlow emulator")
        from forestflow.statistics import p1d as ff_p1d

        from cup1d.theory.projection import fused_projection_Mpc

        k_Mpc = np.asarray(k_Mpc, dtype=float)
        n_batch, n_z, n_k = k_Mpc.shape
        arinyo = self.arinyo(inputs)
        p1d = np.zeros((n_batch, n_z, n_k))
        moments = None
        for index, cosmology in enumerate(cosmologies):
            self.model.set_linear_theory(zs, cosmology=cosmology)
            parameters = {name: np.asarray(values[index], dtype=float)[:, None, None]
                          for name, values in arinyo.items()}
            p1d[index], point_moments = fused_projection_Mpc(
                self.model.model_Arinyo, self.model.linear, zs, k_Mpc[index],
                parameters, ff_p1d._DEFAULT_P1D_K_MU_INTEGRATOR, moments=True)
            if moments is None:
                moments = {name: np.zeros((n_batch, n_z, n_k)) for name in point_moments}
            for name, values in point_moments.items():
                moments[name][index] = values
        return p1d, moments


def set_emulator(emulator_label="lace_mpg", **options):
    """Load a P1D emulator by its cup1d label (see :class:`Emulator`)."""
    return Emulator(emulator_label, **options)


def as_emulator(emulator):
    """``emulator`` itself if it is an :class:`Emulator`, else wrapped in one.

    Lets the theory accept an already built LaCE or ForestFlow emulator (or a
    test double exposing the same methods).
    """
    if isinstance(emulator, Emulator):
        return emulator
    return Emulator(getattr(emulator, "emulator_label", "custom"), model=emulator)
