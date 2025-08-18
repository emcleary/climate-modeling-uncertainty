#!/usr/bin/env python
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import scipy.stats
import sys
from mcmc import MCMC
from gp import GP
from eki import *

if len(sys.argv) != 5:
    print 'Usage: run_mcmc.py eki gp noise_seed y_index'
    sys.exit()

eki_file = sys.argv[1]
gp_file = sys.argv[2]

eki = pickle.load(open(eki_file,'rb'))
gp = pickle.load(open(gp_file,'rb'))

noise_seed = int(sys.argv[3])
y_index = int(sys.argv[4])


if not os.path.exists('mcmc'):
    os.makedirs('mcmc')
output = 'mcmc/mcmc_y_'+str(y_index)+'_seed_'+str(noise_seed)+'.pickle'


p_rh = pickle.load(open('p_uni.pickle','rb'))
p_tau = pickle.load(open('p_ln.pickle','rb'))

if not os.path.isfile(output):
    true_data = eki.g_t
    yi = true_data[y_index]

    theta_t = np.array([p_rh.finv(0.7), p_tau.finv(2*3600.0)])
    mu, cov_gp = gp.predict(theta_t[np.newaxis])
    cov_gp = cov_gp.squeeze()

    u = np.concatenate(eki.u[:10], 0)
    cov_prop = np.cov(u.T)

    param_init = theta_t


    prior = [p_rh, p_tau]
    
    mcmc = MCMC(yi, cov_gp, prior, cov_prop, param_init)
else:
    mcmc = pickle.load(open(output,'rb'))


while len(mcmc.posterior) < 110000:

    param = mcmc.param[None,:]

    # Evaluate the emulator instead of running the full model
    g, _ = gp.predict(param)
    
    mcmc.sampler(g)
    
    if np.mod(len(mcmc.posterior), 1000) == 0:
        print 'Iteration: ', len(mcmc.posterior), ' Acceptance rate: ', mcmc.compute_acceptance_rate()
        pickle.dump(mcmc, open(output,'wb'))
        if len(mcmc.posterior) < 5000:
            mcmc.cov_prop = 1.3*np.cov(mcmc.posterior[-1000:].T)
