#!/usr/bin/env python
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import scipy.stats
import sys


class MCMC:

    def __init__(self, truth, cov_data, priors, cov_prop, param_init):

        self.y = truth
        self.cov = cov_data
        self.cov_inv = np.linalg.inv(self.cov)
        self.priors = priors # List of mappings used
        self.cov_prop = cov_prop

        self.param = param_init
        self.posterior = np.array([param_init])
        self.log_posterior = None
        
    def compute_acceptance_rate(self):
        iterations = self.posterior.shape[0]
        assert iter > 0
        accept = 0
        for i in range(iterations-1):
            idx = i# + self.burnin
            if not np.all(self.posterior[idx] == self.posterior[idx+1]):
                accept += 1

        return float(accept) / iterations

    def log_likelihood(self, g):
        if isinstance(self.y, list): # Essentially product of independent likelihoods
            log_rho = 0.0
            for yi in self.y:
                diff = g - yi
                log_rho += -0.5 * diff.dot(self.cov_inv.dot(diff.T))
        else:
            diff = g - self.y
            log_rho = -0.5 * diff.dot(self.cov_inv.dot(diff.T))
        return log_rho

    def log_prior(self, param=None):
        if param is None:
            param = self.param
        log_rho = 0.0
        for parameter, prior in zip(param, self.priors):
            log_rho += np.log(scipy.stats.norm(prior.mu_z, prior.sig_z).pdf(parameter))
                
        return log_rho

    def proposal(self):
        prop = scipy.stats.multivariate_normal(self.posterior[-1], self.cov_prop).rvs() # input covariance, not sds
        if np.isscalar(prop):
            return np.array([prop])
        else:
            return prop

    def sampler(self, g):

        log_posterior = self.log_likelihood(g) + self.log_prior()

        if self.log_posterior == None:
            self.log_posterior = log_posterior
        else:
            p_accept = np.exp( log_posterior - self.log_posterior )
            if p_accept > np.random.uniform():
                self.posterior = np.append(self.posterior, [self.param], axis=0)
                self.log_posterior = log_posterior
            else:
                self.posterior = np.append(self.posterior, [self.posterior[-1]], axis=0)
                
        # Propose new parameters
        self.param = self.proposal()
