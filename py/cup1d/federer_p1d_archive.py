import numpy as np
import copy
import sys
import os
import json
import matplotlib.pyplot as plt

from . import p1d_archive


class FedererArchiveP1D(p1d_archive.ArchiveP1D):
    """Book-keeping of flux P1D measured in a suite of Federer simulations."""

    def __init__(self,basedir=None,p1d_label=None,skewers_label=None,
                drop_tau_rescalings=False,drop_temp_rescalings=False,
                keep_every_other_rescaling=False,nearest_tau=False,
                max_arxiv_size=None,undersample_z=1,verbose=False,
                no_skewers=False,pick_sim_number=None,drop_sim_number=None,
                z_max=5.,nsamples=None,undersample_cube=1):
        """Load archive from base sim directory and (optional) label
            identifying skewer configuration (number, width)"""

        assert ('LYA_EMU_REPO' in os.environ),'export LYA_EMU_REPO'
        repo=os.environ['LYA_EMU_REPO']

        self.fulldir=repo+basedir
        if p1d_label:
            self.p1d_label=p1d_label
        else:
            self.p1d_label='p1d'
        if skewers_label:
            self.skewers_label=skewers_label
        else:
            self.skewers_label='Ns100_wM0.07'
        self.verbose=verbose
        self.drop_tau_rescalings=drop_tau_rescalings
        self.drop_temp_rescalings=drop_temp_rescalings
        self.nearest_tau=nearest_tau
        self.z_max=z_max
        self.undersample_cube=undersample_cube

        self._load_data(drop_tau_rescalings,drop_temp_rescalings,
                            max_arxiv_size,undersample_z,no_skewers,
                            pick_sim_number,drop_sim_number,
                            keep_every_other_rescaling,
                            z_max,undersample_cube,nsamples)
        
        if nearest_tau:
            self._keep_nearest_tau()

        print('FedererArchiveP1D will call constructor of ArchiveP1D')
        ArchiveP1D.__init__(self,basedir)


    def _load_data(self,drop_tau_rescalings,drop_temp_rescalings,
                            max_arxiv_size,undersample_z,no_skewers,
                            pick_sim_number,drop_sim_number,
                            keep_every_other_rescaling,
                            z_max,undersample_cube,nsamples=None):
        """Setup arxiv by looking at all measured power spectra in sims"""

        # each measured power will have a dictionary, stored here
        self.data=[]

        # read file containing information about latin hyper-cube
        cube_json=self.fulldir+'/latin_hypercube.json'
        with open(cube_json) as json_file:  
            self.cube_data = json.load(json_file)
        if self.verbose:
            print('latin hyper-cube data',self.cube_data)
        if nsamples is None:
            self.nsamples=self.cube_data['nsamples']
        else:
            self.nsamples=nsamples
        if self.verbose:
            print('simulation suite has %d samples'%self.nsamples)

        if pick_sim_number is not None:
            start=pick_sim_number
            self.nsamples=pick_sim_number+1
        else:
            start=0

        # read info from all sims, all snapshots, all rescalings
        for sample in range(start,self.nsamples,undersample_cube):
            if sample is drop_sim_number:
                continue
            # store parameters for simulation pair / model
            sim_params = self.cube_data['samples']['%d'%sample]
            if self.verbose:
                print(sample,'sample has sim params =',sim_params)
            model_dict ={'sample':sample,'sim_param':sim_params}

            # read number of snapshots (should be the same in all sims)
            pair_dir=self.fulldir+'/sim_pair_%d'%sample
            pair_json=pair_dir+'/parameter.json'
            with open(pair_json) as json_file:  
                pair_data = json.load(json_file)
            zs=pair_data['zs']
            Nz=len(zs)
            if self.verbose:
                print('simulation has %d redshifts'%Nz)
                print('undersample_z =',undersample_z)

            # to make lighter emulators, we might undersample redshifts
            for snap in range(0,Nz,undersample_z):       
                if zs[snap]>z_max:
                    continue
                # get linear power parameters describing snapshot
                linP_params = pair_data['linP_zs'][snap]
                snap_p1d_data = {}
                snap_p1d_data['Delta2_p'] = linP_params['Delta2_p']
                snap_p1d_data['n_p'] = linP_params['n_p']
                snap_p1d_data['alpha_p'] = linP_params['alpha_p']
                if 'f' in linP_params:
                    snap_p1d_data['f_p'] = linP_params['f']
                else:
                    snap_p1d_data['f_p'] = linP_params['f_p']
                snap_p1d_data['z']=zs[snap]

                # check if we have extracted skewers yet
                if no_skewers:
                    self.data.append(snap_p1d_data)
                    continue

                # make sure that we have skewers for this snapshot (z < zmax)
                plus_p1d_json=pair_dir+'/sim_plus/{}_{}_{}.json'.format(
                                self.p1d_label,snap,self.skewers_label)
                if not os.path.isfile(plus_p1d_json):
                    if self.verbose:
                        print(plus_p1d_json,'snapshot does not have p1d')
                    continue
                # open file with 1D power measured in snapshot for sim_plus
                with open(plus_p1d_json) as json_file:
                    plus_data = json.load(json_file)
                # open file with 1D power measured in snapshot for sim_minus
                minus_p1d_json=pair_dir+'/sim_minus/{}_{}_{}.json'.format(
                                self.p1d_label,snap,self.skewers_label)
                with open(minus_p1d_json) as json_file: 
                    minus_data = json.load(json_file)

                # number of post-process rescalings for each snapshot
                Npp=len(plus_data['p1d_data'])
                # read info for each post-process
                for pp in range(Npp):
                    # deep copy of dictionary (thread safe, why not)
                    p1d_data = json.loads(json.dumps(snap_p1d_data))
                    k_Mpc = np.array(plus_data['p1d_data'][pp]['k_Mpc'])
                    if len(k_Mpc) != len(minus_data['p1d_data'][pp]['k_Mpc']):
                        print(sample,snap,pp)
                        print(len(k_Mpc),'!=',
                                    len(minus_data['p1d_data'][pp]['k_Mpc']))
                        raise ValueError('different k_Mpc in minus/plus')
                    # average plus + minus stats
                    plus_pp=plus_data['p1d_data'][pp]
                    minus_pp=minus_data['p1d_data'][pp]
                    plus_mF = plus_pp['mF']
                    minus_mF = minus_pp['mF']
                    pair_mF = 0.5*(plus_mF+minus_mF)
                    p1d_data['mF'] = pair_mF 
                    p1d_data['T0'] = 0.5*(plus_pp['sim_T0']+minus_pp['sim_T0'])
                    p1d_data['gamma'] = 0.5*(plus_pp['sim_gamma']
                                            +minus_pp['sim_gamma'])
                    p1d_data['sigT_Mpc'] = 0.5*(plus_pp['sim_sigT_Mpc']
                                            +minus_pp['sim_sigT_Mpc'])
                    # store also scalings used (not present in old versions)
                    if 'sim_scale_T0' in plus_pp:
                        p1d_data['scale_T0'] = plus_pp['sim_scale_T0']
                    if 'sim_scale_gamma' in plus_pp:
                        p1d_data['scale_gamma'] = plus_pp['sim_scale_gamma']
                    # store also filtering length (not present in old versions)
                    if 'kF_Mpc' in plus_pp:
                        p1d_data['kF_Mpc'] = 0.5*(plus_pp['kF_Mpc']
                                                +minus_pp['kF_Mpc'])
                    p1d_data['scale_tau'] = plus_pp['scale_tau']
                    # compute average of < F F >, not <delta delta> 
                    plus_p1d = np.array(plus_pp['p1d_Mpc'])
                    minus_p1d = np.array(minus_pp['p1d_Mpc'])
                    pair_p1d = 0.5*(plus_p1d * plus_mF**2
                                + minus_p1d * minus_mF**2) / pair_mF**2
                    p1d_data['k_Mpc'] = k_Mpc
                    p1d_data['p1d_Mpc'] = pair_p1d
                    self.data.append(p1d_data)                

        if keep_every_other_rescaling:
            if self.verbose: print('will keep every other rescaling in arxiv')
            self._keep_every_other_rescaling()
        if drop_tau_rescalings:
            if self.verbose: print('will drop tau scalings from arxiv')
            self._drop_tau_rescalings()
        if drop_temp_rescalings:
            if self.verbose: print('will drop temperature scalings from arxiv')
            self._drop_temperature_rescalings()

        if max_arxiv_size is not None:
            Ndata=len(self.data)
            if Ndata > max_arxiv_size:
                if self.verbose:
                    print('will keep only',max_arxiv_size,'entries')
                keep=np.random.randint(0,Ndata,max_arxiv_size)
                keep_data=[self.data[i] for i in keep]
                self.data=keep_data

        N=len(self.data)
        if self.verbose:
            print('Arxiv setup, containing %d entries'%len(self.data))

        # create 1D arrays with all entries for a given parameter
        self._store_param_arrays()

        return


    def _store_param_arrays(self):
        """ create 1D arrays with all entries for a given parameter. """

        N=len(self.data)

        # store linear power parameters
        self.Delta2_p=np.array([self.data[i]['Delta2_p'] for i in range(N)])
        self.n_p=np.array([self.data[i]['n_p'] for i in range(N)])
        self.alpha_p=np.array([self.data[i]['alpha_p'] for i in range(N)])
        self.f_p=np.array([self.data[i]['f_p'] for i in range(N)])
        self.z=np.array([self.data[i]['z'] for i in range(N)])

        # store IGM parameters (if present)
        if self.data[0]['mF']:
            self.mF=np.array([self.data[i]['mF'] for i in range(N)])
        if self.data[0]['sigT_Mpc']:
            self.sigT_Mpc=np.array([self.data[i]['sigT_Mpc'] for i in range(N)])
        if self.data[0]['gamma']:
            self.gamma=np.array([self.data[i]['gamma'] for i in range(N)])
        if self.data[0]['kF_Mpc']:
            self.kF_Mpc=np.array([self.data[i]['kF_Mpc'] for i in range(N)])

        return

    def _keep_every_other_rescaling(self):
        """Keep only every other rescaled entry"""

        # select scalings that we want to keep
        keep_tau_scales=np.unique([x['scale_tau'] for x in self.data])[::2]
        keep_T0_scales=np.unique([x['scale_T0'] for x in self.data])[::2]
        keep_gamma_scales=np.unique([x['scale_gamma'] for x in self.data])[::2]

        # keep only entries with correct scalings
        data = [x for x in self.data if (
                            (x['scale_tau'] in keep_tau_scales) &
                            (x['scale_T0'] in keep_T0_scales) &
                            (x['scale_gamma'] in keep_gamma_scales))]

        self.data = data
        return


    def _drop_tau_rescalings(self):
        """Keep only entries with scale_tau=1"""

        data = [x for x in self.data if x['scale_tau']==1.0]
        self.data = data
        return

    def _keep_nearest_tau(self):
        """ Keep only entries with the nearest tau scalings
        Hardcoding this to 0.7 and 1.4 for now, which we used
        in the 200 x 256**3 suite"""

        data = [x for x in self.data if x['scale_tau']==1.0 or x['scale_tau']==0.7 or x['scale_tau']==1.4]
        self.data = data
        return


    def _drop_temperature_rescalings(self):
        """Keep only entries with scale_T0=1 and scale_gamma=1"""

        data = [x for x in self.data if ((x['scale_T0']==1.0)
                                                & (x['scale_gamma']==1.0))]
        self.data = data
        return


    def print_entry(self,entry,keys=['z','Delta2_p','n_p','alpha_p','f_p',
                                    'mF','sigT_Mpc','gamma','kF_Mpc']):
        """Print basic information about a particular entry in the arxiv"""

        if entry >= len(self.data):
            raise ValueError('{} entry does not exist in arxiv'.format(entry))

        data = self.data[entry]
        info='entry = {}'.format(entry)
        for key in keys:
            info += ', {} = {:.4f}'.format(key,data[key])
        print(info)
        return


    def plot_samples(self,param_1,param_2,
                        tau_scalings=True,temp_scalings=True):
        """For parameter pair (param1,param2), plot each point in the arxiv"""

        # mask post-process scalings (optional)
        emu_data=self.data
        Nemu=len(emu_data)
        if not tau_scalings:
            mask_tau=[x['scale_tau']==1.0 for x in emu_data]
        else:
            mask_tau=[True]*Nemu
        if not temp_scalings:
            mask_temp=[(x['scale_T0']==1.0) 
                        & (x['scale_gamma']==1.0) for x in emu_data]
        else:
            mask_temp=[True]*Nemu

        # figure out values of param_1,param_2 in arxiv
        emu_1=np.array([emu_data[i][param_1] for i in range(Nemu) if (
                                                mask_tau[i] & mask_temp[i])])
        emu_2=np.array([emu_data[i][param_2] for i in range(Nemu) if (
                                                mask_tau[i] & mask_temp[i])])
        emu_z=np.array([emu_data[i]['z'] for i in range(Nemu) if (
                                                mask_tau[i] & mask_temp[i])])
        zmin=min(emu_z)
        zmax=max(emu_z)
        plt.scatter(emu_1,emu_2,c=emu_z,s=1,vmin=zmin, vmax=zmax)
        cbar=plt.colorbar()
        cbar.set_label("Redshift", labelpad=+1)
        plt.xlabel(param_1)
        plt.ylabel(param_2)
        plt.show()

        return


    def plot_3D_samples(self,param_1,param_2, param_3,
                        tau_scalings=True,temp_scalings=True):
        """For parameter trio (param1,param2,param3), plot each point in the arxiv"""

        from mpl_toolkits import mplot3d
        # mask post-process scalings (optional)
        emu_data=self.data
        Nemu=len(emu_data)
        if not tau_scalings:
            mask_tau=[x['scale_tau']==1.0 for x in emu_data]
        else:
            mask_tau=[True]*Nemu
        if not temp_scalings:
            mask_temp=[(x['scale_T0']==1.0) 
                        & (x['scale_gamma']==1.0) for x in emu_data]
        else:
            mask_temp=[True]*Nemu

        # figure out values of param_1,param_2 in arxiv
        emu_1=np.array([emu_data[i][param_1] for i in range(Nemu) if (
                                                mask_tau[i] & mask_temp[i])])
        emu_2=np.array([emu_data[i][param_2] for i in range(Nemu) if (
                                                mask_tau[i] & mask_temp[i])])
        emu_3=np.array([emu_data[i][param_3] for i in range(Nemu) if (
                                                mask_tau[i] & mask_temp[i])])

        emu_z=np.array([emu_data[i]['z'] for i in range(Nemu) if (
                                                mask_tau[i] & mask_temp[i])])
        zmin=min(emu_z)
        zmax=max(emu_z)
        fig = plt.figure()
        ax = plt.axes(projection="3d")
        ax.scatter3D(emu_1, emu_2, emu_3, c=emu_z, cmap='brg',s=8)
        ax.set_xlabel(param_1)
        ax.set_ylabel(param_2)
        ax.set_zlabel(param_3)
        plt.show()

        return

    def sub_arxiv_mf(self,min_mf=0.0,max_mf=1.0):
        """ Return copy of arxiv, with entries in a given mean flux range. """

        # make copy of arxiv
        copy_arxiv=copy.deepcopy(self)
        copy_arxiv.min_mf=min_mf
        copy_arxiv.max_mf=max_mf

        print(len(copy_arxiv.data),'initial entries')

        # select entries in a given mean flux range
        new_data=[d for d in copy_arxiv.data if (
                                        d['mF'] < max_mf and d['mF'] > min_mf)]

        if self.verbose:
            print('use %d/%d entries'%(len(new_data),len(self.data)))

        # store new sub-data
        copy_arxiv.data=new_data

        # re-create 1D arrays with all entries for a given parameter
        copy_arxiv._store_param_arrays()

        return copy_arxiv


    def get_param_values(self,param,tau_scalings=True,temp_scalings=True):
        """ Return values for a given parameter, including rescalings or not."""

        N=len(self.data)
        # mask post-process scalings (optional)
        if not tau_scalings:
            mask_tau=[x['scale_tau']==1.0 for x in self.data]
        else:
            mask_tau=[True]*N
        if not temp_scalings:
            mask_temp=[(x['scale_T0']==1.0) & (x['scale_gamma']==1.0) for x in self.data]
        else:
            mask_temp=[True]*N

        # figure out values of param in arxiv
        values=np.array([self.data[i][param] for i in range(N) if (
                                                mask_tau[i] & mask_temp[i])])

        return values
