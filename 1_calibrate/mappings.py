import numpy as np

# EKI can seems to work best when parameters are on an infinite domain.
# These classes help transform domains to meet this requirement.
# This can be interpreted as a "physical" domain of parameters used
# in the climate models, and a "computational" domain used in the EKI algorithm.
#
# x: Parameters in physical space
# z: Parameters in computational space

class Normal:
    # MIGHT WANT FOR MCMC IF NO PRIORS ARE GIVEN???
    def __init__(self):
        self.mu_z = 0
        self.sig_z = 1 

class Uniform:

    def __init__(self, xlim=[0,1]):

        # Sample z -- use in EKI
        # z in (-infty, infty)
        self.mu_z  = 0
        self.sig_z = 1

        # Convert to x -- use in model
        # x in [xmin, xmax]
        self.xmin = xlim[0]
        self.xmax = xlim[1]

    # Mappings
    def f(self, z):
        return (self.xmax-self.xmin)/(1 + np.exp(-z)) + self.xmin
    
    def finv(self, x):
        return -np.log((self.xmax-self.xmin)/(x-self.xmin) - 1)

    # Sample
    def sample_x(self, J):
        return np.random.uniform(self.xmin, self.xmax, size=J)

    def sample_z(self, J):
        return np.random.normal(self.mu_z, scale=self.sig_z, size=J)


class LogNormal:

    def __init__(self, mu=1, sig=1):

        # x > 0
        self.mu_x  = mu
        self.sig_x = sig
        
        # Sample z -- use in EKI
        # z in (-infty, infty)
        mu_z, sig_z = self.ln2n()
        self.mu_z  = mu_z
        self.sig_z = sig_z

    # Mappings
    def f(self, z):
        return np.exp(z)

    def finv(self, x):
        return np.log(x)

    # Convert statistics
    def ln2n(self):
        a = np.log(self.mu_x)
        b = np.log(self.mu_x**2 + self.sig_x**2)
        mu_z  =  2*a - 0.5*b
        var_z = -2*a +     b
        return mu_z, np.sqrt(var_z)

    def n2ln(self):
        a = self.mu_z + 0.5*self.sig_z**2
        mu_x  = np.exp(  a)
        var_x = np.exp(2*a) * (np.exp(self.sig_z**2) - 1)
        return mu_x, np.sqrt(var_x)

    # Sample
    def sample_x(self, J):
        return np.random.lognormal(self.mu_x, sigma=self.sig_x, size=J)

    def sample_z(self, J):
        return np.random.normal(self.mu_z, scale=self.sig_z, size=J)

