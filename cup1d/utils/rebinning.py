import numpy as np


def get_bin_coverage(xmin_o, xmax_o, xmin_n, xmax_n):
    """Trick to accelerate rebinning"""
    # check out https://stcorp.github.io/harp/doc/html/algorithms/regridding.html
    cover = np.zeros((len(xmin_n), len(xmin_o)))
    for jj in range(len(xmin_n)):
        cover[jj] = np.fmax(
            (np.fmin(xmax_o, xmax_n[jj]) - np.fmax(xmin_o, xmin_n[jj]))
            / (xmax_o - xmin_o),
            0,
        )
    return cover


class Rebinning(object):
    """Class for rebinning

    Only implemented in k_kms for now, but could be extended to z

    """

    def __init__(self, dict_data, k_rebin_factor=1):

        self.k_kms = {}
        self.zs = {}
        self.cover = {}
        self.sum_cover = {}

        for key in dict_data:
            self.k_kms[key] = []  # new k_kms
            self.cover[key] = []  # to accelerate rebinning
            self.sum_cover[key] = []  # to accelerate rebinning

            data = dict_data[key]
            self.zs[key] = data.z

            if k_rebin_factor == 1:
                # Do nothing for a factor of one; retain the original k grid.
                for iz in range(len(self.zs[key])):
                    self.k_kms[key].append(data.k_kms[iz])
                    self.cover[key].append(
                        np.ones((len(data.k_kms[iz]), len(data.k_kms[iz])))
                    )
                    self.sum_cover[key].append(np.sum(self.cover[key][iz], axis=1))
            else:
                for iz in range(len(self.zs[key])):
                    nelem = len(data.k_kms[iz]) * k_rebin_factor
                    _kms_reb = np.linspace(
                        data.k_kms_min[iz][0] * 0.95,
                        data.k_kms_max[iz][-1] * 1.05,
                        nelem,
                    )
                    self.k_kms[key].append(_kms_reb)
                    xmin_o = _kms_reb - 0.5 * (_kms_reb[1] - _kms_reb[0])
                    xmax_o = _kms_reb + 0.5 * (_kms_reb[1] - _kms_reb[0])

                    _cover = get_bin_coverage(
                        xmin_o,
                        xmax_o,
                        data.k_kms_min[iz],
                        data.k_kms_max[iz],
                    )
                    self.cover[key].append(_cover)
                    self.sum_cover[key].append(np.sum(_cover, axis=1))

    def rebinning(self, key, Pk_kms_newk):
        """For rebinning Pk predictions"""
        Pk_kms_origk = []
        for iz in range(len(self.zs[key])):
            _Pk_kms = (
                np.sum(
                    self.cover[key][iz] * Pk_kms_newk[iz][np.newaxis, :],
                    axis=1,
                )
                / self.sum_cover[key][iz]
            )
            Pk_kms_origk.append(_Pk_kms)
        return Pk_kms_origk

    def rebinning_batch(self, key, Pk_kms_newk):
        """Rebin a batch of P1D predictions without looping over walkers.

        ``Pk_kms_newk`` is a list over redshift bins. Item ``iz`` has shape
        ``(n_batch, n_k_fine[iz])`` and the returned item has shape
        ``(n_batch, n_k_data[iz])``. Keeping a list over redshift preserves
        the ragged k grids used by the observational data.
        """

        if len(Pk_kms_newk) != len(self.zs[key]):
            raise ValueError(
                f"expected {len(self.zs[key])} redshift predictions for {key}; "
                f"got {len(Pk_kms_newk)}"
            )
        rebinned = []
        n_batch = None
        for iz, prediction in enumerate(Pk_kms_newk):
            prediction = np.asarray(prediction, dtype=float)
            if prediction.ndim != 2:
                raise ValueError(
                    "batched prediction must have shape (n_batch, n_k); "
                    f"redshift {iz} has shape {prediction.shape}"
                )
            if prediction.shape[1] != self.cover[key][iz].shape[1]:
                raise ValueError(
                    f"redshift {iz} has {prediction.shape[1]} k bins, expected "
                    f"{self.cover[key][iz].shape[1]}"
                )
            if n_batch is None:
                n_batch = prediction.shape[0]
            elif prediction.shape[0] != n_batch:
                raise ValueError("all redshift predictions must share n_batch")
            weights = self.cover[key][iz] / self.sum_cover[key][iz][:, None]
            rebinned.append(prediction @ weights.T)
        return rebinned


# def rebinning(key, zs, Pk_kms_finek):
#     """For rebinning Pk predictions"""
#     Pk_kms_origk = []
#     # _Pk_kms_finek = np.atleast_1d(Pk_kms_finek)
#     for iz in range(len(zs)):
#         indz = np.argmin(np.abs(self.data.z - zs[iz]))
#         _Pk_kms = (
#             np.sum(
#                 self.rebin[key]["cover"][indz] * Pk_kms_finek[iz][np.newaxis, :],
#                 axis=1,
#             )
#             / self.rebin[key]["sum_cover"][indz]
#         )
#         Pk_kms_origk.append(_Pk_kms)
#     return Pk_kms_origk
