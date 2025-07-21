import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
import numpy as np
from cup1d.likelihood.input_pipeline import Args
from cup1d.likelihood.pipeline import Pipeline
from cup1d.utils.utils import get_path_repo


def main():
    args = Args(emulator_label="CH24_mpgcen_gpr", training_set="Cabayol23")
    # fit_type = "global"
    fit_type = "andreu2"
    args.set_baseline(
        fit_type=fit_type, fix_cosmo=True, P1D_type="DESIY1_QMLE3"
    )
    path_out = os.path.join(
        os.path.dirname(get_path_repo("cup1d")),
        "data",
        "out_DESI_DR1",
        args.P1D_type,
        args.fit_type,
        args.emulator_label,
    )

    pip = Pipeline(args, out_folder=path_out)
    p0 = pip.fitter.like.sampling_point_from_parameters().copy()
    pip.run_profile(args)


if __name__ == "__main__":
    main()
