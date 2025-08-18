import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from gp import GP

# Load training data
eki = pickle.load(open('eki.pickle','rb'))

sample_idx = -1
X = np.concatenate(eki.u[:5], axis=0)
Y = np.concatenate(eki.g[:5], axis=0)

# Regression stuff
normalizer = True
kernel = 'RBF'

# SVD stuff
svd = True
truth = eki.g_t
cov = eki.cov

# Ensure we only use the mean of the truth
if truth.ndim == 2:
    truth = truth.mean(0)

# Sparse GP stuff
sparse = True
n_induce = 50

# GP model
gp = GP(X, Y, normalizer=normalizer, kernel=kernel, \
        svd=svd, truth=truth, cov=cov, \
        sparse=sparse, n_induce=n_induce)
gp.initialize_models()
pickle.dump(gp, open('gp.pickle','wb'))
