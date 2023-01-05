import os
import configargparse
import time
# our own modules
from lace.emulator import gp_emulator
from cup1d.data import mock_data
from cup1d.likelihood import likelihood
from cup1d.likelihood import emcee_sampler

os.environ["OMP_NUM_THREADS"] = "1"

parser = configargparse.ArgumentParser()
parser.add_argument('--timeout', type=float, required=True,
        help='Stop chain after these many hours')
parser.add_argument('--subfolder', type=str, default='mock',
        help='Subdirectory to save chain file in')
parser.add_argument('--emu_type', type=str, default='polyfit',
        help='k_bin or polyfit emulator')
parser.add_argument('--kmax_Mpc', type=float, default=8,
        help='Maximum k to train emulator')
parser.add_argument('--z_max', type=float, default=4.5,
        help='Maximum redshift')
parser.add_argument('--n_igm', type=int, default=2,
        help='Number of free parameters for IGM model')
parser.add_argument('--no_igm', action='store_true',
        help='Do not vary IGM parameters at all')
parser.add_argument('--cosmo_fid_label', type=str, default='default',
        help='Fiducial cosmology to use (default,truth)')
parser.add_argument('--burn_in', type=int, default=200,
        help='Number of burn in steps')
parser.add_argument('--prior_Gauss_rms', type=float, default=0.5,
        help='Width of Gaussian prior')
parser.add_argument('--data_label', type=str, default='Chabanier2019',
        help='Data covariance to use, Chabanier2019 or QMLE_Ohio')
parser.add_argument('--rootdir', type=str, default=None,
        help='Root directory containing chains')
parser.add_argument('--extra_p1d_label', type=str, default=None,
        help='Which extra p1d data covmats to use (e.g., Karacayli2022)')
parser.add_argument('--free_cosmo_params', nargs="+",
        help='List of cosmological parameters to sample')
args = parser.parse_args()

print('--- print options from parser ---')
print(args)
print("----------")
print(parser.format_help())
print("----------")
print(parser.format_values()) 
print("----------")

if args.rootdir:
    rootdir=args.rootdir
    print('set input rootdir',rootdir)
else:
    assert ('CUP1D_PATH' in os.environ),'Define CUP1D_PATH variable'
    rootdir=os.environ['CUP1D_PATH']+'/chains/'
    print('use default rootdir',rootdir)

# compile list of free parameters
if args.free_cosmo_params:
    free_parameters=args.free_cosmo_params
else:
    free_parameters=['As','ns']

# do not add IGM parameters (testing)
if args.no_igm:
    print('running without IGM parameters')
else:
    print('using {} parameters for IGM model'.format(args.n_igm))
    for i in range(args.n_igm):
        for par in ["tau","sigT_kms","gamma","kF"]:
            free_parameters.append('ln_{}_{}'.format(par,i))
print('free parameters',free_parameters)

# set up an emulator
emu=gp_emulator.GPEmulator(emu_type=args.emu_type,
                        kmax_Mpc=args.kmax_Mpc)

# generate mock P1D measurement
data=mock_data.Mock_P1D(emulator=emu,
                        data_label=args.data_label,
                        zmax=args.z_max)

# check if we want to include high-resolution data
if args.extra_p1d_label:
    extra_p1d_data=mock_data.Mock_P1D(emulator=emu,
                        data_label=args.extra_p1d_label,
                        zmax=args.z_max)
else:
    extra_p1d_data=None

# create likelihood object from data and emulator
like=likelihood.Likelihood(data=data,emulator=emu,
                        free_param_names=free_parameters,
                        prior_Gauss_rms=args.prior_Gauss_rms,
                        cosmo_fid_label=args.cosmo_fid_label,
                        extra_p1d_data=extra_p1d_data)

# pass likelihood to sampler
sampler = emcee_sampler.EmceeSampler(like=like,
                        subfolder=args.subfolder,
                        rootdir=rootdir)

# print free parameters
for p in sampler.like.free_params:
    print(p.name,p.value,p.min_value,p.max_value)

# cannot call self.log_prob using multiprocess.pool
def log_prob(theta):
    return sampler.like.log_prob_and_blobs(theta)

# actually run the sampler
start = time.time()
sampler.run_sampler(burn_in=args.burn_in,max_steps=10000000,
            log_func=log_prob,parallel=True,timeout=args.timeout)
end = time.time()
multi_time = end - start
print("Sampling took {0:.1f} seconds".format(multi_time))

# store results (skip plotting when running at NERSC)
sampler.write_chain_to_file(residuals=True,plot_nersc=True,
            plot_delta_lnprob_cut=50)
