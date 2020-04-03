import numpy as np
import os
import json

from cup1d.archive import p1d_archive


class ExampleArchiveP1D(p1d_archive.ArchiveP1D):
    """Book-keeping of flux P1D measured in a suite of Example simulations."""

    def __init__(self,basedir=None,prefix='p1d',suffix='Ns256_wM0.05',
                verbose=False):
        """Setup archive from simulations in basedir folder.
            Filenames are prefix_snap_suffix.json, where snap = snapshot number.
        """

        # set verbosity level
        self.verbose=verbose

        # read data from suite of simulations
        data=self._read_data(basedir,prefix,suffix)

        print('ExampleArchiveP1D will call constructor of ArchiveP1D')
        p1d_archive.ArchiveP1D.__init__(self,data)


    def _read_data(self,basedir,prefix,suffix):
        """Look at all sims in the basedir folder, and read P1D measured.
        Inputs:
            - basedir: folder containing the suite of simulations.
            - prefix/suffix: p1d files should be named prefix_x_suffix.json,
                where x is the snapshot number.
        """

        self.verbose=True

        # each measured power will have a dictionary, stored here
        data=[]

        # read file containing information about latin hyper-cube
        cube_json=basedir+'/latin_hypercube.json'
        with open(cube_json) as json_file:  
            self.cube_data = json.load(json_file)
        if self.verbose:
            print('latin hyper-cube data',self.cube_data)

        self.nsamples=self.cube_data['nsamples']
        if self.verbose:
            print('simulation suite has %d samples'%self.nsamples)

        # read info from all sims, all snapshots, all rescalings
        for sample in range(self.nsamples):
            # store parameters for simulation pair / model
            sim_params = self.cube_data['samples']['%d'%sample]
            if self.verbose:
                print(sample,'sample has sim params =',sim_params)
            model_dict ={'sample':sample,'sim_param':sim_params}

            # read number of snapshots (should be the same in all sims)
            pair_dir=basedir+'/sim_pair_%d'%sample
            pair_json=pair_dir+'/parameter.json'
            with open(pair_json) as json_file:  
                pair_data = json.load(json_file)
            zs=pair_data['zs']
            Nz=len(zs)
            if self.verbose:
                print('simulation has %d redshifts'%Nz)

            for snap in range(Nz):
                # get linear power parameters describing snapshot
                linP_params = pair_data['linP_zs'][snap]
                snap_p1d_data = {}
                snap_p1d_data['Delta2_p'] = linP_params['Delta2_p']
                snap_p1d_data['n_p'] = linP_params['n_p']
                snap_p1d_data['alpha_p'] = linP_params['alpha_p']
                snap_p1d_data['f_p'] = linP_params['f_p']
                snap_p1d_data['z']=zs[snap]

                # make sure that we have skewers for this snapshot 
                plus_p1d_json=pair_dir+'/sim_plus/{}_{}_{}.json'.format(
                                prefix,snap,suffix)
                if not os.path.isfile(plus_p1d_json):
                    if self.verbose:
                        print(plus_p1d_json,'snapshot does not have p1d')
                    continue

                # open file with 1D power measured in snapshot for sim_plus
                with open(plus_p1d_json) as json_file:
                    plus_data = json.load(json_file)
                # open file with 1D power measured in snapshot for sim_minus
                minus_p1d_json=pair_dir+'/sim_minus/{}_{}_{}.json'.format(
                                prefix,snap,suffix)
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
                    data.append(p1d_data)                

        N=len(data)
        if self.verbose:
            print('Arxiv setup, containing %d entries'%len(data))

        return data

