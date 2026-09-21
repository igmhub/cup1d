"""Prior domains in the compressed linear-power parameter plane."""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class EmulatorPriorPlotter:
    """Construct and display MPG and Nyx emulator prior domains.

    The domains are obtained by mapping the rectangular ``A_s``--``n_s``
    ranges used for each emulator into the ``Delta2_star``--``n_star`` plane.
    The boundary is a concave hull of that mapped grid, matching the prior
    construction formerly implemented in the variations notebook.
    """

    z_star: float = 3.0
    kp_kms: float = 0.009
    grid_size: int = 50
    alpha: float = 1.0
    boundaries: dict = field(default_factory=dict, init=False)
    samples: dict = field(default_factory=dict, init=False)
    simulation_points: dict = field(default_factory=dict, init=False)

    _RANGES = {
        "mpg": {
            "As": (1.026970414602751e-09, 3.4382747625135757e-09),
            "ns": (0.6922930599534987, 1.2746555165892985),
        },
        "nyx": {
            "As": (8.062826641805509e-10, 4.658124996088607e-09),
            "ns": (0.6465761801374897, 1.3448138385530823),
        },
    }

    def compute_priors(self):
        """Compute the two mapped prior domains and return their boundaries."""

        self._set_fiducial_cosmology()
        for emulator, ranges in self._RANGES.items():
            self.samples[emulator] = self._map_primordial_grid(ranges)
            self.boundaries[emulator] = self._concave_hull(
                self.samples[emulator]
            )
            self.simulation_points[emulator] = self._simulation_points(emulator)
        return self.boundaries

    def plot_priors(self, ax=None, show_simulations=True):
        """Plot the MPG and Nyx prior boundaries and simulation cosmologies."""

        if not self.boundaries:
            self.compute_priors()
        if ax is None:
            import matplotlib.pyplot as plt

            _, ax = plt.subplots(figsize=(7, 5))

        styles = {"mpg": ("C0", "MPG"), "nyx": ("C1", "Nyx")}
        for emulator, (color, label) in styles.items():
            boundary = self.boundaries[emulator]
            ax.plot(boundary[:, 0], boundary[:, 1], color=color, label=label)
            if show_simulations:
                points = self.simulation_points[emulator]
                ax.scatter(points[:, 0], points[:, 1], color=color, s=22)

        ax.set_xlabel(r"$\Delta^2_\star$")
        ax.set_ylabel(r"$n_\star$")
        ax.legend(frameon=False)
        return ax

    def _set_fiducial_cosmology(self):
        """Set the CAMB-derived reference used for the fast mapping."""

        from lace.cosmo import camb_cosmo
        from cup1d.theory.camb import CAMBModel

        cosmology = camb_cosmo.get_cosmology(
            H0=67.66,
            mnu=0.0,
            omch2=0.119,
            ombh2=0.0224,
            omk=0.0,
            As=2.105e-09,
            ns=0.9665,
            nrun=0.0,
            pivot_scalar=0.05,
            w=-1,
        )
        model = CAMBModel(
            zs=[self.z_star],
            cosmo=cosmology,
            z_star=self.z_star,
            kp_kms=self.kp_kms,
            fast_camb=False,
        )
        star_params = model.get_linP_params()
        self._fiducial = {
            "As": 2.105e-09,
            "ns": 0.9665,
            "nrun": 0.0,
            **star_params,
        }
        self._kp_mpc = self.kp_kms * model.dkms_dMpc(self.z_star)

    def _map_primordial_grid(self, ranges):
        """Map a regular primordial-spectrum grid to star parameters."""

        amplitudes = np.linspace(*ranges["As"], self.grid_size)
        tilts = np.linspace(*ranges["ns"], self.grid_size)
        amplitude_grid, tilt_grid = np.meshgrid(amplitudes, tilts)
        log_pivot_ratio = np.log(self._kp_mpc / 0.05)
        delta_tilt = tilt_grid.ravel() - self._fiducial["ns"]
        log_amplitude_ratio = np.log(
            amplitude_grid.ravel() / self._fiducial["As"]
        )
        delta2_star = self._fiducial["Delta2_star"] * np.exp(
            log_amplitude_ratio + delta_tilt * log_pivot_ratio
        )
        n_star = self._fiducial["n_star"] + delta_tilt
        return np.column_stack((delta2_star, n_star))

    def _simulation_points(self, emulator):
        """Return the cosmologies included in the corresponding simulation set."""

        from cup1d.theory.cosmology import set_cosmo

        cosmologies = set_cosmo(f"{emulator}_0", return_all=True)
        points = []
        for label, cosmology in cosmologies.items():
            if emulator == "mpg":
                include = label[-1].isdigit() or label == "mpg_central"
            else:
                include = (
                    (label[-1].isdigit() and label != "accel2")
                    or label == "nyx_central"
                ) and not label.endswith("14")
            if include:
                star = cosmology["star_params"]
                points.append((star["Delta2_star"], star["n_star"]))
        return np.asarray(points)

    def _concave_hull(self, points):
        """Return the exterior coordinates of the alpha shape of ``points``."""

        try:
            import alphashape
            from shapely.geometry import MultiPolygon, Polygon
        except ImportError as error:
            raise ImportError(
                "Computing emulator prior boundaries requires alphashape."
            ) from error

        shape = alphashape.alphashape(points, self.alpha)
        if isinstance(shape, Polygon):
            return np.asarray(shape.exterior.coords)
        if isinstance(shape, MultiPolygon):
            largest = max(shape.geoms, key=lambda polygon: polygon.area)
            return np.asarray(largest.exterior.coords)
        raise ValueError("The alpha shape did not produce a polygonal boundary.")
