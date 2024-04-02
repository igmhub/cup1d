# mpiexec -n 4 python sam_sim.py --emulator_label Pedersen21 --data_label mpg_central --igm_label mpg_central --cosmo_label mpg_central --n_igm 2  --cov_label Chabanier2019 --verbose --parallel

import os
from cup1d.likelihood.input_pipeline import parse_args, SamplerPipeline


def sam_sim(args):
    """Sample the posterior distribution for a of a mock

    Parameters
    ----------
    args : Namespace
        Command line arguments. See cup1d.likelihood.input_pipeline.parse_args()
    """

    pip = SamplerPipeline(args)
    pip.run_sampler()


if __name__ == "__main__":
    args = parse_args()
    if args.parallel:
        os.environ["OMP_NUM_THREADS"] = "1"
    sam_sim(args)
