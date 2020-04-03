import os
from getdist import loadMCSamples

def planck_chains_dir(release):
    """Given a Planck data release (year, integer), return the full path
        to the folder where the chains are stored. 
        It will look for the environmental variable PLANCK_CHAINS"""
    root_dir=os.environ['PLANCK_CHAINS']
    if release==2013:
        return root_dir+'COM_CosmoParams_fullGrid_R1.10/'
    elif release==2015:
        return root_dir+'COM_CosmoParams_fullGrid_R2.00/'
    elif release==2018:
        return root_dir+'COM_CosmoParams_fullGrid_R3.01/'
    else:
        raise ValueError('wrong Planck release',release)


def get_planck_results(release,data,lya_data=None):
    """Load results from Planck, for a given data release and data combination.
    Inputs:
        - release (integer): 2013, 2015 or 2018
        - data (string): data combination, e.g., plikHM_TT_lowl_lowE
    Outputs: 
        - dictionary with relevant information
    """
    analysis={}
    analysis['release']=release
    analysis['root_dir']=planck_chains_dir(analysis['release'])
    # specify analysis and chain name
    analysis['model']='base_mnu'
    analysis['data']=data
    analysis['dir_name']=analysis['root_dir']+'/'+analysis['model']+'/'+analysis['data']+'/'
    # specify Lya chain
    analysis['lya_data']=lya_data
    if lya_data is None:
        analysis['chain_name']=analysis['model']+'_'+analysis['data']
    else:
        analysis['chain_name']=analysis['model']+'_'+analysis['data']+'_'+analysis['lya_data']
    print('chain name =',analysis['chain_name'])
    analysis['samples'] = loadMCSamples(analysis['dir_name']+analysis['chain_name'])
    analysis['parameters']=analysis['samples'].getParams()
    return analysis


def get_planck_2013():
    """Load results from Planck 2013, planck_lowl_lowLike_highL"""
    return get_planck_results(2013,'planck_lowl_lowLike_highL')


def get_planck_2015():
    """Load results from Planck 2015, plikHM_TT_WMAPTEB"""
    return get_planck_results(2015,'plikHM_TT_WMAPTEB')


def get_planck_2018(lya_data=None):
    """Load results from Planck 2018, plikHM_TT_lowl_lowE"""
    return get_planck_results(2018,'plikHM_TT_lowl_lowE',lya_data)


def get_planck_2018_BAO(lya_data=None):
    """Load results from Planck 2018 + BAO, plikHM_TT_lowl_lowE_BAO"""
    return get_planck_results(2018,'plikHM_TT_lowl_lowE_BAO',lya_data)

