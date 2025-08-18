import numpy as np
import matplotlib.pyplot as plt

class EKI:

    # y = g(u) + N(0, cov)
    def __init__(self, parameters, truth, cov, mu=None, sigma=None, T=None):

        # Check inputs
        assert (parameters.ndim == 2), \
            'EKI init: parameters must be 2d array, num_ensembles x num_parameters'
        # assert (truth.ndim == 1), 'EKI init: truth must be 1d array'
        # assert (cov.ndim == 2), 'EKI init: covariance must be 2d array'
        # assert (truth.size == cov.shape[0] and truth.size == cov.shape[1]),\
        #     'EKI init: truth and cov are not the correct sizes'
        
        # Truth statistics
        self.g_t = truth
        self.cov = cov

        # Parameters
        self.u = parameters[np.newaxis]

        # Ensemble size, parameter size
        self.J, self.p = parameters.shape

        # Size of statistics
        self.n_obs = truth.size
        
        # Store observations during iterations
        if truth.ndim == 1:
            self.g = np.empty((0,self.J)+truth.shape)
        else:
            self.g = np.empty((0,self.J)+(truth.shape[1],))

        # Error
        self.error = np.empty(0)

        # Number of timesteps
        self.T = T

        # Prior stats for eks
        self.mu = mu
        self.sigma = sigma
        
    # Parameters corresponding to minimum error.
    # Returns mean and standard deviation.
    def get_u(self):
        return self.u[-1]

    # Minimum error
    def get_error(self):
        try:
            idx = self.error.argmin()
            return self.error[idx]
        except ValueError:
            print('No errors computed.')

    # Compute error using mean of ensemble model evaluations.
    def compute_error(self, iteration=-1):
        if self.g_t.ndim == 1:
            diff = self.g_t - self.g[iteration].mean(0)
        else:
            diff = self.g_t.mean(0) - self.g[iteration].mean(0)
        try:
            error = diff.dot(np.linalg.solve(self.cov, diff))
        except np.linalg.LinAlgError:
            error = diff.dot(diff) / diff.size
        self.error = np.append(self.error, error)
        
    # g: data, i.e. g(u), with shape (num_ensembles, num_elements)
    def update(self, g):
        
        u = np.copy(self.u[-1])
        g_t = self.g_t
        cov = self.cov
        
        # Ensemble size
        J = self.J
        
        # Sizes of u and p
        us = u[0].size
        ps = g[0].size
        
        # Ensemble statistics
        u_bar = np.zeros(us)
        p_bar = np.zeros(ps)
        c_up = np.zeros((us, ps))
        c_pp = np.zeros((ps, ps))
        
        for j in range(J):
            
            u_hat = u[j]
            p_hat = g[j]
            
            # Means
            u_bar += u_hat
            p_bar += p_hat
            
            # Covariance matrices
            c_up += np.tensordot(u_hat, p_hat, axes=0)
            c_pp += np.tensordot(p_hat, p_hat, axes=0)
            
        u_bar = u_bar / J
        p_bar = p_bar / J
        c_up  = c_up  / J - np.tensordot(u_bar, p_bar, axes=0)
        c_pp  = c_pp  / J - np.tensordot(p_bar, p_bar, axes=0)
        
        # Update u
        noise = np.array([np.random.multivariate_normal(np.zeros(ps), cov) for _ in range(J)])
        y = g_t + noise
        tmp = np.linalg.solve(c_pp + cov, np.transpose(y-g))
        u += c_up.dot(tmp).T

        # Store parameters and observations
        self.u = np.append(self.u, [u], axis=0)
        self.g = np.append(self.g, [g], axis=0)

        return

    # EKI update using real data samples rather than summary statistics
    # g: data, i.e. g(u), with shape (num_ensembles, num_elements)
    def update_with_data(self, g):
        
        u = np.copy(self.u[-1])
        g_t = self.g_t
        cov = self.cov
        
        # Ensemble size
        J = self.J
        
        # Sizes of u and p
        us = u[0].size
        ps = g[0].size
        
        # Ensemble statistics
        u_bar = np.zeros(us)
        p_bar = np.zeros(ps)
        c_up = np.zeros((us, ps))
        c_pp = np.zeros((ps, ps))
        
        for j in range(J):
            
            u_hat = u[j]
            p_hat = g[j]
            
            # Means
            u_bar += u_hat
            p_bar += p_hat
            
            # Covariance matrices
            c_up += np.tensordot(u_hat, p_hat, axes=0)
            c_pp += np.tensordot(p_hat, p_hat, axes=0)
            
        u_bar = u_bar / J
        p_bar = p_bar / J
        c_up  = c_up  / J - np.tensordot(u_bar, p_bar, axes=0)
        c_pp  = c_pp  / J - np.tensordot(p_bar, p_bar, axes=0)
        
        # Update u
        # noise = np.array([np.random.multivariate_normal(np.zeros(ps), cov) for _ in range(J)])
        # y = g_t + noise
        idx = np.random.choice(np.arange(len(g_t)), J)
        y = g_t[idx]
        tmp = np.linalg.solve(c_pp + cov, np.transpose(y-g))
        u += c_up.dot(tmp).T

        # Store parameters and observations
        self.u = np.append(self.u, [u], axis=0)
        self.g = np.append(self.g, [g], axis=0)

        return
    

    # Discrete with timestep, timestep term included in the covinv term
    def update_discrete(self, g):

        # CHANGE MY INPUTS TO MATCH ALFREDO'S
        U0 = self.u[-1].T
        y_obs = self.g_t
        Gamma = self.cov
        Geval = g.T
        Jnoise = np.linalg.cholesky(Gamma)
        

	# For ensemble update
	eta   = np.random.normal(0, 1, [self.n_obs, self.J])
	Umean = U0.mean(axis = 1)[:, np.newaxis]
        
	E = Geval - Geval.mean(axis = 1)[:,np.newaxis]
	R = Geval - y_obs[:,np.newaxis]
        
	Cpp = (1./self.J) * np.matmul(E, E.T)
	Cup = (1./self.J) * np.matmul(U0 - Umean, E.T)

        hk = 1./self.T
	D =  (1.0/self.J) * np.matmul(E.T, np.linalg.solve(Gamma + hk * Cpp, R))
        
            
	dU = - hk * np.matmul(U0 - Umean, D)
	dW = np.sqrt(hk) * np.matmul(Cup, np.linalg.solve( hk * Cpp + Gamma,
				                           np.matmul(Jnoise, eta)))
        
	Uk = U0 + dU + dW
        
        # UPDATE MY SELF
        self.u = np.append(self.u, [Uk.T], axis=0)
        self.g = np.append(self.g, [Geval.T], axis=0)

        return
        
    # CONINUOUS EKI -- cov is just gamma, timestep term --> 0
    def update_continuous(self, g):

        # CHANGE MY INPUTS TO MATCH ALFREDO'S
        U0 = self.u[-1].T
        y_obs = self.g_t
        Gamma = self.cov
        Geval = g.T
        Jnoise = np.linalg.cholesky(Gamma)

        # For ensemble update
        eta   = np.random.normal(0, 1, [self.n_obs, self.J])
        Umean = U0.mean(axis = 1)[:, np.newaxis]

        E = Geval - Geval.mean(axis = 1)[:,np.newaxis]
        R = Geval - y_obs[:,np.newaxis]

        Cpp = (1./self.J) * np.matmul(E, E.T)
        Cup = (1./self.J) * np.matmul(U0 - Umean, E.T)

        D =  (1.0/self.J) * np.matmul(E.T, np.linalg.solve(Gamma, R))
        # Adaptive timestep
        hk = 1./np.linalg.eigvals(D).real.max()

        dU = - hk * np.matmul(U0 - Umean, D)
        dW = np.sqrt(hk) * np.matmul(Cup, np.linalg.solve(Gamma,
                        np.matmul(Jnoise, eta)))
        Uk = U0 + dU + dW
        
        # UPDATE MY SELF
        self.u = np.append(self.u, [Uk.T], axis=0)
        self.g = np.append(self.g, [Geval.T], axis=0)

        return
    
    # EKS
    def update_eks(self, g):

        # CHANGE MY INPUTS TO MATCH ALFREDO'S
        U0 = self.u[-1].T
        y_obs = self.g_t.mean(0)
        Gamma = self.cov
        Geval = g.T
        Jnoise = np.linalg.cholesky(Gamma)

        # For ensemble update
        E = Geval - Geval.mean(axis = 1)[:,np.newaxis]
        R = Geval - y_obs[:,np.newaxis]
        D =  (1.0/self.J) * np.matmul(E.T, np.linalg.solve(Gamma, R))

        hk = 1./(np.linalg.norm(D) + 1e-8)

        Umean = U0.mean(axis = 1)[:, np.newaxis]
        Ucov  = np.cov(U0) + 1e-8 * np.identity(self.p)

        Ustar_ = np.linalg.solve(np.eye(self.p) + hk * np.linalg.solve(self.sigma.T, Ucov.T).T,
                U0 - hk * np.matmul(U0 - Umean, D)  + hk * np.matmul(Ucov, np.linalg.solve(self.sigma, self.mu)))
        Uk     = (Ustar_ + np.sqrt(2*hk) * np.matmul( np.linalg.cholesky(Ucov),
                np.random.normal(0, 1, [self.p, self.J])))

        # UPDATE MY SELF
        self.u = np.append(self.u, [Uk.T], axis=0)
        self.g = np.append(self.g, [Geval.T], axis=0)

        return

    def plot_error(self):
        x = np.arange(self.error.size)
        plt.semilogy(x, self.error)
        plt.xlabel('Iteration')
        plt.ylabel(r'$\parallel y - g(u)\parallel_\Gamma^2$')
