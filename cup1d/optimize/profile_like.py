import numpy as np


def run_profile(fitter, file_out, mle_cosmo_cen, input_pars, nelem=10):
    # mpg, np -2.35, -2.25
    # mpg, delta2p 0.1, 0.8
    err_cosmo = {"Delta2_star": 0.04, "n_star": 0.018}
    x = np.linspace(-2, 2, nelem)
    xgrid, ygrid = np.meshgrid(x, x)
    xgrid = xgrid.reshape(-1) * err_cosmo["Delta2_star"]
    ygrid = ygrid.reshape(-1) * err_cosmo["n_star"]

    chi2_arr = np.zeros(len(xgrid))
    blind_cosmo = np.zeros((len(xgrid), 2))

    for ii in range(len(xgrid)):
        print(
            ii,
            len(xgrid),
            xgrid[ii] / err_cosmo["Delta2_star"],
            ygrid[ii] / err_cosmo["n_star"],
        )

        target = fitter.apply_unblinding(mle_cosmo_cen)
        target["Delta2_star"] += xgrid[ii]
        target["n_star"] += ygrid[ii]

        blind_cosmo[ii, 0] = mle_cosmo_cen["Delta2_star"] + xgrid[ii]
        blind_cosmo[ii, 1] = mle_cosmo_cen["n_star"] + ygrid[ii]

        fitter.like.theory.rescale_fid_cosmo(target)

        # check whether new fiducial cosmology is within priors
        if np.isfinite(fitter.like.get_chi2(input_pars)) == False:
            print("skipping", ii, blind_cosmo[ii])
            continue

        fitter.run_minimizer(
            fitter.like.minus_log_prob, p0=input_pars, restart=True, nsamples=0
        )

        print("\n", "\n", fitter.mle_chi2)
        chi2_arr[ii] = fitter.mle_chi2

        out_dict = {
            "chi2": chi2_arr,
            "blind_cosmo": blind_cosmo,
        }

        np.save(file_out, out_dict)

    return chi2_arr, blind_cosmo
