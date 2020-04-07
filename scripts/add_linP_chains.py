import numpy as np
from cup1d.planck import planck_chains
from cup1d.planck import add_linP_params

# load Planck 2018 chain
planck2018=planck_chains.get_planck_2018()

# reduce sice of chain, at least while testing
samples=planck2018['samples']
thinning=20000
samples.thin(thinning)
Nsamp=len(samples.weights)
print('will use %d samples'%Nsamp)

linP_params=[]
for i in range(Nsamp):
    verbose=(i%2==0)
    if verbose: print('sample point',i)
    params=samples.getParamSampleDict(i)
    linP=add_linP_params.get_linP_params(params,verbose=verbose)
    linP_params.append(linP)

# setup numpy arrays with linP parameters
linP_DL2_star=np.array([linP_params[i]['Delta2_star'] for i in range(Nsamp)])
linP_n_star=np.array([linP_params[i]['n_star'] for i in range(Nsamp)])
linP_alpha_star=np.array([linP_params[i]['alpha_star'] for i in range(Nsamp)])
linP_f_star=np.array([linP_params[i]['f_star'] for i in range(Nsamp)])
linP_g_star=np.array([linP_params[i]['g_star'] for i in range(Nsamp)])

# add new derived linP parameters 
samples.addDerived(linP_DL2_star,'linP_DL2_star',
            label='Ly\\alpha \\, \\Delta_\\ast')
samples.addDerived(linP_n_star,'linP_n_star',
            label='Ly\\alpha \\, n_\\ast')
samples.addDerived(linP_alpha_star,'linP_alpha_star',
            label='Ly\\alpha \\, \\alpha_\\ast')
samples.addDerived(linP_f_star,'linP_f_star',
            label='Ly\\alpha \\, f_\\ast')
samples.addDerived(linP_g_star,'linP_g_star',
            label='Ly\\alpha \\, g_\\ast')

# get basic statistics for the new parameters
param_means=np.mean(samples.samples,axis=0)
param_vars=np.var(samples.samples,axis=0)
print('DL2_star mean = {} +/- {}'.format(param_means[88],np.sqrt(param_vars[88])))
print('n_star mean = {} +/- {}'.format(param_means[89],np.sqrt(param_vars[89])))
print('alpha_star mean = {} +/- {}'.format(param_means[90],np.sqrt(param_vars[90])))
print('f_star mean = {} +/- {}'.format(param_means[91],np.sqrt(param_vars[91])))
print('g_star mean = {} +/- {}'.format(param_means[92],np.sqrt(param_vars[92])))

# store new chain to file
new_root_name=planck2018['dir_name']+planck2018['chain_name']+'_linP'
if (thinning > 1.0):
    new_root_name+='_'+str(thinning)
    print('new root name',new_root_name)
    samples.saveAsText(root=new_root_name,make_dirs=True)
