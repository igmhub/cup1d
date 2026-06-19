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

    def __init__(self, dict_data, rebin_k=1):

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

            if rebin_k == 1:
                # do not do anything if rebin_k == 1, just store the original k_kms
                for iz in range(len(self.zs[key])):
                    self.k_kms[key].append(data.k_kms[iz])
                    self.cover[key].append(
                        np.ones((len(data.k_kms[iz]), len(data.k_kms[iz])))
                    )
                    self.sum_cover[key].append(np.sum(self.cover[key][iz], axis=1))
            else:
                for iz in range(len(self.zs[key])):
                    nelem = len(data.k_kms[iz]) * rebin_k
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
