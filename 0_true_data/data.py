import numpy as np
from netCDF4 import Dataset
import sys
import scipy.interpolate

# Purpose of this class is to load and merge data from parallel output,
# along with some postprocessing.

class Data:

    global gc_ratio
    global T0
    global e0
    global hlv
    global rvgas
    global rdgas
    
    gc_ratio = 0.621972 # ratio of gas constants for dry air and water vapor
    T0    = 273.16  # K
    e0    = 610.78  # Pa
    hlv   = 2.5E6   # J/kg
    rvgas = 461.50  # J/kg/deg (water vapor gas constant)
    rdgas = 287.04  # J/kg/deg (dry air gas constant)

    def __init__(self, files):

        self.files = files
        
        rootgrplist = []
        for file in files:
            rootgrplist.append(Dataset(file))
        self.rootgrplist = rootgrplist

        # Load/merge dimensions
        self.time = rootgrplist[0].variables['time'][:]   # days
        self.pfull = rootgrplist[0].variables['pfull'][:] # hPa
        self.phalf = rootgrplist[0].variables['phalf'][:] # hPa
        self.lon = rootgrplist[0].variables['lon'][:]
        self.lat = self.load_variable('lat')

        # Load/merge variables
        self.psfc = self.load_variable('ps')    # Pa
        self.temp = self.load_variable('temp')  # K
        self.shum = self.load_variable('sphum') # kg/kg

        # Compute data dimensions
        self.sigma = self.pfull / self.phalf[-1]

        # del self.rootgrplist

    # Merges across latitude dimension by default.
    def load_variable(self, varname, dim = 'lat'):

        tmp = []
        for rootgrp in self.rootgrplist:
            tmp.append(rootgrp.variables[varname][:])

        var = self.rootgrplist[0].variables[varname]
        idx = var.dimensions.index(dim)

        return np.concatenate(tmp, idx)

        

    def zonal_average(self, data):

        nsig  = np.shape(data)[1]
        nlon  = np.shape(data)[3]

        field = data * self.psfc[:,np.newaxis,:,:]
        field = field.sum(3) # sum on longitude

        return field / nlon / (100*self.phalf[nsig])

    
    def time_average(self, data, idx=0):
        return data.mean(idx)


    def compute_sat_mixing_ratio(self):

        # Simple exponential form for saturation vapor pressure
        sat_vapor_pressure = e0 * np.exp(-hlv/rvgas * (1.0/self.temp - 1.0/T0))

        pfull = self.pfull[np.newaxis,:,np.newaxis,np.newaxis]
        sat_mixing_ratio = (gc_ratio * sat_vapor_pressure) \
            / (100*pfull - sat_vapor_pressure)
        
        return sat_mixing_ratio

    
    def compute_rhum(self):

        mixing_ratio = self.shum / (1.0 - self.shum)
        sat_mr = self.compute_sat_mixing_ratio()
        
        rhum = mixing_ratio/sat_mr * (1 + (rvgas/rdgas) * sat_mr) \
                    / (1 + (rvgas/rdgas) * mixing_ratio)

        return rhum

    
    
    def extract(self, data, case, lat0 = 0, sig0 = 0.5):

        if case == 'scalar':
            lat = lat0
            sig = sig0
        elif case == 'latitude':
            lat = self.lat
            sig = sig0
        elif case == 'sigma':
            lat = lat0
            sig = self.sigma
        else:
            print 'Cannot use '+case+' in compute_g!'



        if data.ndim == 3:
            g = []
            for d in data: # loop over time
                f = scipy.interpolate.interp2d(self.lat, self.sigma, d)
                g.append(f(lat, sig))
            g = np.asarray(g)

        elif data.ndim == 2:
            f = scipy.interpolate.interp2d(self.lat, self.sigma, data)
            g = f(lat, sig)

        else:
            print 'Input has wrong shape!'
            sys.exit()

        return g
        
    def compute_precip(self):
        conv = self.load_variable('convection_rain')
        cond = self.load_variable('condensation_rain')
        return conv+cond
