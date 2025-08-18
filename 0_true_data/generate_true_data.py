import numpy as np
from scipy.stats import truncnorm
import os.path
import os
import pickle
from utilities import *
import sys
from data import Data

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

# Input directory
# Return truth
def load_truth(dir):
    files = ['/'.join([dir, file]) for file in os.listdir(dir)] 
    files.sort()
    return Data(files)


def main():

    ##########################################################
    # 1: CHANGE AS NEEDED
    ##########################################################

    # param_dir = 'parameters'
    # error_file = 'error.txt'
    # data_file = 'data.pickle'
    # eki_file = 'eki.pickle'
    truth_dir = 'truth_alpha_1.0'
    days = 30 # number of days to average for the truth (choose same
              # value as GCM ouptuts for EKI)
    dataperday = 4 # GCM outputs per day
    truth_day0 = 3000 # day to start from when computing averages from truth

    # dump_data = False

    precip_threshold = 1.1547e-5
    precip_percentile = 90.0
    precip_units = 86400.0 # s/day

    # if not os.path.exists(param_dir):
    #     os.makedirs(param_dir)

    # Subset of forward model data used
    # extraction = 'scalar'
    extraction = 'latitude' # extracts along latitude at specified altitude
    # extraction = 'sigma' # extracts along sigma at specified latitude

    p_uni = pickleLoad('p_uni.pickle')
    p_ln = pickleLoad('p_ln.pickle')

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

        #####################
        # RELATIVE HUMIDITY #
        #####################
        rhum_all = []
        for dir in dirs:
            truth = load_truth(dir)
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

        #################
        # PRECIPITATION #
        #################

        # Compute precipitation (convection + condensation) (NOTE: all precip, NOT daily averages)
        precip_all = []
        for dir in dirs:
            truth = load_truth(dir)
            precip = truth.compute_precip()
            precip_all.append(precip)

        del truth, precip
        precip_all = np.concatenate(precip_all, axis=0)
        precip_daily_averages = compute_temporal_averages(precip_all, dataperday, 1)
        out['precip daily averages'] = precip_daily_averages * precip_units

        # Compute zonal daily averages
        precip_zonals = precip_daily_averages.mean(axis=2)

        # Compute monthly averages
        precip_monthly_averages = compute_temporal_averages(precip_zonals, 1, days)
        precip_monthly_averages *= precip_units

        ############
        # EXTREMES #
        ############

        # Threshold
        threshold = []
        nlat = precip_daily_averages.shape[1]
        for i in range(nlat):
            # Filter out nonrainy days
            precip = precip_daily_averages[:,i,:].ravel()
            # Remove days with "no rain"
            mask = precip > precip_threshold
            precip = precip[mask]
            # Calculate threshold at  percentile
            t = np.percentile(precip, precip_percentile)
            threshold.append(t)
        threshold = np.asarray(threshold)
        pickleDump('data/threshold_wo_noise.pickle', threshold)
        out['threshold'] = threshold

        # Probabilities
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
        out['extremes'] = extremes

        ##################################
        # DUMP TRUTH WITHOUT ADDED NOISE #
        ##################################

        # Merge all data
        truth_all = np.concatenate([rhum_means, precip_monthly_averages, extremes], axis=1)
        pickleDump('data/truth_wo_noise.pickle', truth_all)

        out['truth'] = truth_all
        pickleDump('data/truth_all_wo_noise.pickle', out)

        #############################################
        # TRANSFORM DATA, ADD NOISE, TRANSFORM BACK #
        #############################################

        # Relative humidity
        z_rhum = p_uni.finv(rhum_means)
        cov_z_rhum = np.diag(np.var(z_rhum, axis=0, ddof=1))
        noise = np.random.multivariate_normal(np.zeros(z_rhum.shape[1]), 10*cov_z_rhum, z_rhum.shape[0])
        rhum_means = p_uni.f(z_rhum+noise)

        # Daily precipitation
        nlat = precip_daily_averages.shape[1]
        for i in range(nlat):
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

        #################
        # PRECIPITATION #
        #################

        out['precip daily averages'] = precip_daily_averages * precip_units

        # Compute zonal daily averages
        precip_zonals = precip_daily_averages.mean(axis=2)

        # Compute monthly averages
        precip_monthly_averages = compute_temporal_averages(precip_zonals, 1, days)
        precip_monthly_averages *= precip_units

        ############
        # EXTREMES #
        ############

        # Threshold
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
        pickleDump('data/threshold.pickle', threshold)
        out['threshold'] = threshold

        # Probabilities
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
        out['extremes'] = extremes
    
        ##################################
        # DUMP TRUTH WITHOUT ADDED NOISE #
        ##################################

        # Merge all data
        truth_all = np.concatenate([rhum_means, precip_monthly_averages, extremes], axis=1)
        pickleDump('data/truth.pickle', truth_all)

        out['truth'] = truth_all
        pickleDump('data/truth_all.pickle', out)

    
main()
