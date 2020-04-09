import numpy as np
import os
import time
from cup1d.planck import planck_chains
from cup1d.planck import add_linP_params

# point to original Planck chains
root_dir=os.environ['PLANCK_CHAINS']
# load Planck chain
model='base_mnu'
release=2018
if release==2013:
    data=None
    #data='WMAP'
    planck=planck_chains.get_planck_2013(model=model,data=data,
                                root_dir=root_dir,linP_tag=None)
elif release==2015:
    data=None
    planck=planck_chains.get_planck_2015(model=model,data=data,
                                root_dir=root_dir,linP_tag=None)
elif release==2018:
    data=None
    #data='plikHM_TTTEEE_lowl_lowE_BAO'
    planck=planck_chains.get_planck_2018(model=model,data=data,
                                root_dir=root_dir,linP_tag=None)
else:
    print('wrong relase',release)
    exit()

# do not compute (f_star,g_star), only linP at z=3
z_evol=True

# reduce sice of chain, at least while testing
samples=planck['samples']
thinning=10
samples.thin(thinning)
Nsamp,Npar=samples.samples.shape
print('Thinned chains have {} samples and {} parameters'.format(Nsamp,Npar))

# print in total 100 updates
print_every=int(Nsamp/100)+1
print('will print every',print_every)

# time execution
start_time = time.time()

linP_params=[]
for i in range(Nsamp):
    verbose=(i%print_every==0)
    if verbose: print('sample point',i)
    # actually, let's not print than that
    verbose=False
    params=samples.getParamSampleDict(i)
    linP=add_linP_params.get_linP_params(params,z_evol=z_evol,verbose=verbose)
    if verbose: print('linP params',linP)
    linP_params.append(linP)

end_time = time.time()
print('ellapsed time',end_time - start_time)

# setup numpy arrays with linP parameters
linP_DL2_star=np.array([linP_params[i]['Delta2_star'] for i in range(Nsamp)])
linP_n_star=np.array([linP_params[i]['n_star'] for i in range(Nsamp)])
linP_alpha_star=np.array([linP_params[i]['alpha_star'] for i in range(Nsamp)])
if z_evol:
    linP_f_star=np.array([linP_params[i]['f_star'] for i in range(Nsamp)])
    linP_g_star=np.array([linP_params[i]['g_star'] for i in range(Nsamp)])

# add new derived linP parameters 
samples.addDerived(linP_DL2_star,'linP_DL2_star',
            label='Ly\\alpha \\, \\Delta_\\ast')
samples.addDerived(linP_n_star,'linP_n_star',
            label='Ly\\alpha \\, n_\\ast')
samples.addDerived(linP_alpha_star,'linP_alpha_star',
            label='Ly\\alpha \\, \\alpha_\\ast')
if z_evol:
    samples.addDerived(linP_f_star,'linP_f_star',
            label='Ly\\alpha \\, f_\\ast')
    samples.addDerived(linP_g_star,'linP_g_star',
            label='Ly\\alpha \\, g_\\ast')

# get basic statistics for the new parameters
param_means=np.mean(samples.samples,axis=0)
param_vars=np.var(samples.samples,axis=0)
print('DL2_star mean = {} +/- {}'.format(param_means[Npar],np.sqrt(param_vars[Npar])))
print('n_star mean = {} +/- {}'.format(param_means[Npar+1],np.sqrt(param_vars[Npar+1])))
print('alpha_star mean = {} +/- {}'.format(param_means[Npar+2],np.sqrt(param_vars[Npar+2])))
if z_evol:
    print('f_star mean = {} +/- {}'.format(param_means[Npar+3],np.sqrt(param_vars[Npar+3])))
    print('g_star mean = {} +/- {}'.format(param_means[Npar+4],np.sqrt(param_vars[Npar+4])))

# store new chain to file
new_root_name=planck['dir_name']+planck['chain_name']
if z_evol:
    new_root_name += '_zlinP'
else:
    new_root_name += '_linP'
if (thinning > 1.0):
    new_root_name+='_'+str(thinning)
print('new root name',new_root_name)
samples.saveAsText(root=new_root_name,make_dirs=True)
