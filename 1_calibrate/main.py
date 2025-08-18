import numpy as np
from scipy.stats import truncnorm
import os.path
import os
import pickle
from utilities import *
import sys
from data import Data
from eki import *
from mappings import Uniform, LogNormal

# Assumes index 0 is time
def compute_statistics(data):
    mean = np.mean(data, axis=0)
    cov = np.cov(data.T)
    if cov.ndim == 1: cov = np.array([[cov]])
    return mean, cov

# daily averages -- assumes index 0 is time
# Ex: GCM outputs 4 sets of data per day, and want monthly average:
#     data_averages = compute_temporal_averages(data, 4, 30)
def compute_temporal_averages(data, dataperday, ndays):
    nsamples = data.shape[0] / (dataperday * ndays)
    d = np.asarray([data[dataperday*ndays*i:dataperday*ndays*(i+1)].mean(axis=0) \
                    for i in range(nsamples)])
    return d

# precip: zonal averages (timesteps, lat) or (ensemble, lat)
# thredhold: extreme threshold array (lat)
# precip_threshold: must excede to be considered a rainy day
def compute_extremes(precip, threshold, precip_threshold=1.1547e-5):
    extremes = []
    minprob = 1e-6
    # MUST CHECK THAT I LOOP OVER LATITUDES, p SHOULD BE (TIME, LON) or (TIME*LON)
    nlon = threshold.size
    for i in range(nlon):
        p = precip[:,i,:].ravel()
        t = threshold[i]
        mask = p > precip_threshold
        precip_filtered = p[mask]
        # Calculate probability of exceeding the threshold
        prob = 0
        if len(precip_filtered) > 0:
            prob = float(sum(precip_filtered > t)) / len(precip_filtered)
        if prob < minprob:
            prob = minprob
        extremes.append(prob)
    return np.asarray(extremes)


def main():

    ##########################################################
    # 1: CHANGE AS NEEDED
    ##########################################################

    param_dir = 'parameters'
    error_file = 'error.txt'
    data_file = 'data.pickle'
    eki_file = 'eki.pickle'
    truth_dir = 'truth_alpha_1.0'
    days = 30 # number of days to average for the truth (choose same
              # value as GCM ouptuts for EKI)
    dataperday = 4 # GCM outputs per day
    truth_day0 = 3000 # day to start from when computing averages from truth

    dump_data = False

    precip_threshold = 1.1547e-5
    precip_percentile = 90.0
    precip_units = 86400.0 # s/day

    if not os.path.exists(param_dir):
        os.makedirs(param_dir)

    # Subset of forward model data used
    # extraction = 'scalar'
    extraction = 'latitude' # extracts along latitude at specified altitude
    # extraction = 'sigma' # extracts along sigma at specified latitude

    p_uni = Uniform([0,1])
    p_ln = LogNormal(mu=3600.0*12, sig=3600.0*12) # I might need to include min/max in this (here min = 0.5 hrs)

    pickleDump('p_uni.pickle',p_uni)
    pickleDump('p_ln.pickle',p_ln)


    ##########################################################
    # 2: EKI truth / observations / measurements
    ##########################################################

    if not os.path.exists('data'):
        os.makedirs('data')
    
    if not os.path.isfile('data/truth.pickle'):

        # Locate truth directory
        dir = os.getcwd().split('fms-idealized')[0] + 'fms_tmp/' + truth_dir
        # truth_dir = '/'.join(tmploc + ['fms_tmp','enkf_truth'])
        truth_list = os.listdir(dir)
        if 'mppnccombine.ifc' in truth_list: truth_list.remove('mppnccombine.ifc')
        if 'exe.fms' in truth_list: truth_list.remove('exe.fms')
        if len(truth_list) != 1:
            print 'Must only be 1 case in truth directory '+truth_dir
            sys.exit()
        # dir = '/'.join([dir, truth_list[0]])
        dirs = ['/'.join([dir, truth_dir]) for truth_dir in truth_list]
        dir = '/'.join([dir,truth_dir,'output','combine'])
        dirs = ['/'.join([dir,data]) for data in os.listdir(dir)]
        dirs = sorted(dirs, key=lambda d: int(d.split('day')[-1].split('h')[0])) 
        while truth_day0 > int(dirs[0].split('day')[-1].split('h')[0]):
            dirs = dirs[1:]

        # Compute relative humidity
        rhum_all = []
        for dir in dirs:
            # Load data
            files = ['/'.join([dir, file]) for file in os.listdir(dir)] 
            files.sort()
            truth = Data(files)
            rhum = truth.compute_rhum()
            rhum = truth.zonal_average(rhum)
            rhum_all.append(truth.extract(rhum, extraction))

        out = {}
        out['lat'] = truth.lat
        out['lon'] = truth.lon
        out['sigma'] = truth.sigma
        out['psfc'] = truth.psfc

        del truth, rhum
        rhum_all = np.concatenate(rhum_all, axis=0)
        ntimesteps = rhum_all.shape[0]
        nlat = rhum_all.shape[1]
        out['rhum'] = rhum_all

        # Compute rhum temporal means
        rhum_means = compute_temporal_averages(rhum_all, dataperday, days)
        del rhum_all

        # Compute precipitation (convection + condensation) (NOTE: all precip, NOT daily averages)
        precip_all = []
        for dir in dirs:
            # Load data
            files = ['/'.join([dir, file]) for file in os.listdir(dir)] 
            files.sort()
            truth = Data(files)
            precip = truth.compute_precip()
            precip_all.append(precip)

        del truth, precip
        precip_all = np.concatenate(precip_all, axis=0)
        precip_daily_averages = compute_temporal_averages(precip_all, dataperday, 1)
        out['precip daily averages (without noise)'] = precip_daily_averages * precip_units

        #############################################
        # TRANSFORM DATA, ADD NOISE, TRANSFORM BACK #
        #############################################

        # Relative humidity
        z_rhum = p_uni.finv(rhum_means)
        cov_z_rhum = np.diag(np.var(z_rhum, axis=0, ddof=1))
        noise = np.random.multivariate_normal(np.zeros(z_rhum.shape[1]), 10*cov_z_rhum, z_rhum.shape[0])
        rhum_means = p_uni.f(z_rhum+noise)

        # Daily precipitation
        nlot = precip_daily_averages.shape[1]
        for i in range(nlot):
            # Filter out nonrainy days
            precip = precip_daily_averages[:,i,:].ravel()
            # Calculate noise
            z_precip = p_ln.finv(precip)
            mask = z_precip != -np.inf
            m_precip = z_precip[mask]
            noise_stddev = np.sqrt(0.01*np.var(m_precip, ddof=1))
            noise = np.random.normal(0, noise_stddev, z_precip.shape)
            precip = p_ln.f(z_precip+noise)
            # Overwrite data without noise
            shape = precip_daily_averages[:,i,:].shape
            precip_daily_averages[:,i,:] = precip.reshape(shape)

        #############################################

        out['precip daily averages (with noise)'] = precip_daily_averages * precip_units

        # Compute precipitation threshold
        file = 'data/threshold.pickle'
        threshold = []
        nlot = precip_daily_averages.shape[1]
        for i in range(nlot):
            # Filter out nonrainy days
            precip = precip_daily_averages[:,i,:].ravel()
            # Remove days with "no rain"
            mask = precip > precip_threshold
            precip = precip[mask]
            # Calculate threshold at  percentile
            t = np.percentile(precip, precip_percentile)
            threshold.append(t)
        threshold = np.asarray(threshold)
        pickleDump(file, threshold)
        out['threshold (with noisy precip)'] = threshold

        # Compute precipitation extremes -- NOT extreme daily averages
        extremes = []
        ntimesteps = precip_daily_averages.shape[0]
        for i in range(0, ntimesteps/days):
            extremes.append(
                compute_extremes(
                    precip_daily_averages[days*i:days*(i+1)], 
                    threshold, 
                    precip_threshold=precip_threshold
                )
            )
        extremes = np.asarray(extremes)
        out['extremes (noisy precip threshold and data)'] = extremes

        # Compute zonal daily averages
        precip_zonals = precip_daily_averages.mean(axis=2)

        # Compute monthly averages
        precip_monthly_averages = compute_temporal_averages(precip_zonals, 1, days)
        precip_monthly_averages *= precip_units

        # Merge all data
        truth_all = np.concatenate([rhum_means, precip_monthly_averages, extremes], axis=1)
        pickleDump('data/truth.pickle', truth_all)

        out['truth'] = truth_all
        pickleDump('data/truth_all.pickle', out)
        del out
    
    if not os.path.isfile(eki_file):
        truth_all = pickleLoad('data/truth.pickle')
        # Sigmoid of RH
        truth_all[:,:32] = p_uni.finv(truth_all[:,:32])
        # Log of extremes
        truth_all[:,64:] = p_ln.finv(truth_all[:,64:])
        
        # Compute statistics
        mean, cov = compute_statistics(truth_all)

        # Load initial parameters
        initial_parameters = read_param(param_dir)

        # Transform parameters
        z_rh = p_uni.finv(initial_parameters[:,0])
        z_tau = p_ln.finv(initial_parameters[:,1])
        p0 = np.vstack([z_rh, z_tau]).T
        
        # Initialize EKI object
        mu = np.array([p_uni.mu_z, p_ln.mu_z])[:,np.newaxis]
        sig = np.diag(np.array([p_uni.sig_z, p_ln.sig_z]))
        eki_obj = EKI(p0, truth_all, cov, mu, sig)
        pickleDump(eki_file, eki_obj)
        
    else:
        eki_obj = pickleLoad(eki_file)

    threshold = pickleLoad('data/threshold.pickle')

    ##########################################################
    # 3: Compute/Postprocess/Extract data from ensemble GCM
    ##########################################################

    # Ensemble directories
    homedir = os.getcwd().split('fms-idealized')[0]
    expdir = os.getcwd().split('exp')[1].split('/')[1]
    ensdir = expdir+'_ens'
    dir = homedir + 'fms_tmp/' + ensdir
    ens_list = os.listdir(dir)
    if 'mppnccombine.ifc' in ens_list: ens_list.remove('mppnccombine.ifc')
    if 'exe.fms' in ens_list: ens_list.remove('exe.fms')
    ens_list.sort(key=lambda x: int(x.split('_')[-1]))
    ens_dirs = ['/'.join([dir, d]) for d in ens_list]

    # Load data objects
    ens = []
    g = []
    for dir in ens_dirs:
        files = file_list(dir)
        data = Data(files)

        g_tmp = np.array([])

        rhum = data.compute_rhum()
        rhum = data.zonal_average(rhum)
        rhum = data.time_average(rhum)
        rhum = data.extract(rhum, extraction)
        rhum = p_uni.finv(rhum)

        precip = data.compute_precip()
        precip_daily_averages = compute_temporal_averages(precip, dataperday, 1)

        # Compute zonal daily averages
        precip_zonals = precip_daily_averages.mean(axis=2) # Zonal averages

        # Compute precipitation extremes
        extremes = compute_extremes(precip_daily_averages, threshold, precip_threshold=precip_threshold)
        extremes = p_ln.finv(extremes)

        # Time average over whole ensemble
        precip = precip_zonals.mean(axis=0) # Time average
        precip *= precip_units

        # Store data
        g.append(np.concatenate([rhum, precip, extremes], axis=0))

        if dump_data:
            
            # Output
            out = {}
            out['time'] = data.time
            out['lon'] = data.lon
            out['lat'] = data.lat
            out['sigma'] = data.sigma
            out['psfc'] = data.psfc
            out['temp'] = data.temp
            out['shum'] = data.shum
            out['rhum'] = rhum
            out['precip'] = precip
                
            tmp = '/'.join(['data', dir.split('/')[-1]])
            if not os.path.exists(tmp):
                os.makedirs(tmp)
                
            file = '/'.join([tmp, files[0].split('/')[-1].split('.')[0]])
            pickleDump(file, out)
        
    # Merge g
    g = np.concatenate([g], axis=0)

    ##########################################################
    # 4: EKI iteration and dump
    ##########################################################

    # Run iteration of EKI
    eki_obj.update_with_data(g)

    # Dump
    pickleDump(eki_file, eki_obj)

    ##########################################################
    # 5: Write parameters/errors
    ##########################################################

    # Parameters
    u = eki_obj.u[-1]

    # Transform parameters for model
    x_rh = u[:,0]
    x_rh = p_uni.f(u[:,0])
    x_tau = p_ln.f(u[:,1])
    x = np.vstack([x_rh, x_tau]).T
    writeout_ensemble(param_dir, x)
    
    # Errors
    eki_obj.compute_error()
    err = eki_obj.error[-1]
    writeout(error_file, np.array([err]))#, ['||y-G(u)||_cov'])
    
main()
