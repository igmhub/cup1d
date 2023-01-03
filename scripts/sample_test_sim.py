import os
import configargparse
import time
# our own modules
from lace.emulator import gp_emulator
from lace.emulator import p1d_archive
from cup1d.data import data_MPGADGET
from cup1d.likelihood import likelihood
from cup1d.likelihood import emcee_sampler

os.environ["OMP_NUM_THREADS"] = "1"

parser = configargparse.ArgumentParser()
parser.add_argument('--timeout', type=float, required=True,
        help='Stop chain after these many hours')
parser.add_argument('--subfolder', type=str, default='cup1d',
        help='Subdirectory to save chain file in')
parser.add_argument('--emu_type', type=str, default='polyfit',
        help='k_bin or polyfit emulator')
parser.add_argument('--sim_label', type=str, default='central',
        help='Which sim to use as mock data')
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
parser.add_argument('--data_cov_factor', type=float, default=1.0,
        help='Factor to multiply the data covariance by')
parser.add_argument('--data_cov_label', type=str, default='Chabanier2019',
        help='Data covariance to use, Chabanier2019 or PD2013')
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

basedir='/lace/emulator/sim_suites/Australia20/'

if args.rootdir:
    rootdir=args.rootdir
    print('set input rootdir',rootdir)
else:
    assert ('P1D_FORECAST' in os.environ),'Define P1D_FORECAST variable'
    rootdir=os.environ['P1D_FORECAST']+'/chains/'
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

# check if sim_label is part of the training set, and remove it
if args.sim_label.isdigit():
    drop_sim_number=int(args.sim_label)
    print('dropping simulation from training set',drop_sim_number)
else:
    drop_sim_number=None
    print('using test simulation',args.sim_label)

# generate mock P1D measurement
data=data_MPGADGET.P1D_MPGADGET(basedir=basedir,
                        sim_label=args.sim_label,
			            zmax=args.z_max,
                        data_cov_factor=args.data_cov_factor,
                        data_cov_label=args.data_cov_label,
                        polyfit=(args.emu_type=='polyfit'))

# set up emulator training data
archive=p1d_archive.archiveP1D(basedir=basedir,z_max=args.z_max,
                        drop_sim_number=drop_sim_number,
                        drop_tau_rescalings=True,
                        drop_temp_rescalings=True)

# set up an emulator
emu=gp_emulator.GPEmulator(basedir,train=True,
                        passarchive=archive,
                        emu_type=args.emu_type,
                        kmax_Mpc=args.kmax_Mpc,
                        asymmetric_kernel=True,
                        rbf_only=True)

# check if we want to include high-resolution data
if args.extra_p1d_label:
    extra_p1d_data=data_MPGADGET.P1D_MPGADGET(basedir=basedir,
                        sim_label=args.sim_label,
                        zmax=args.z_max,
                        data_cov_label=args.extra_p1d_label,
                        data_cov_factor=1.0,
                        polyfit=(args.emu_type=='polyfit'))
else:
    extra_p1d_data=None

# create likelihood object from data and emulator
like=likelihood.Likelihood(data=data,emulator=emu,
                        free_param_names=free_parameters,
                        prior_Gauss_rms=args.prior_Gauss_rms,
                        cosmo_fid_label=args.cosmo_fid_label,
                        extra_p1d_data=extra_p1d_data,
                        verbose=False)

# pass likelihood to sampler
sampler = emcee_sampler.EmceeSampler(like=like,verbose=False,
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
