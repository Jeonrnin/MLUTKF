"""Lorenz 1996 model (with zonally varying damping)
Lorenz E., 1996. Predictability: a problem partly solved. In
Predictability. Proc 1995. ECMWF Seminar, 1-18."""
import numpy as np
from enkf import get_truncated_normal

class L96:
    def __init__(self, enkf_method=0, nem=1, nem_sclfct=1, sp_dist=0, sp_dist_nrml=0, nrml_bmt=None, dt=0.05, N=40, ndim_loc=1, blend=1., F=8., deltaF=0., Fcorr=1., proc_q=0.01, en=0, lamda=0., rs=None):
        self.enkf_method = enkf_method
        self.nem = nem                                                                                  # the number of ensemble members of EnKF                                                                           
        self.nem_sclfct = nem_sclfct                                                                    # scaling factor of ensemble size for UTKF, LUTKF, SUTKF and SLUTKF
        self.sp_dist = sp_dist                                                                          # distribution of ensemble (sigma point) for UTKF, LUTKF, SUTKF and SLUTKF (0: uniform distribution, 1: normal distribution, 2: normal distribution using box muller transform)
        self.sp_dist_nrml = sp_dist_nrml                                                                # normal distribution of ensemble (sigma point) for UTKF and SUTKF (when self.sp_dist = 1) (0: each element, 1: entire element)
        self.nrml_bmt = nrml_bmt                                                                        # normal distribution using box muller transform
        self.dt = dt                                                                                    # model interval
        self.N = N                                                                                      # the number of grid points along a latitude circle (each grid has one state variable). Moreover, N is dimension of L96 model (global model).
        self.ndim_loc = ndim_loc                                                                        # local dimension (each grid has one state variable)    
        self.blend = blend                                                                              # [optional] advection diffusion parameter
        self.F = F                                                                                      # forcing
        self.deltaF = deltaF                                                                            # [optional] scale parameter of gamma distribution
        self.Fcorr = Fcorr                                                                              # [optional] forcing correlation parameter
        self.proc_q = proc_q                                                                            # process error variance
        self.en = en                                                                                    # n for SUTKF and SLUTKF
        self.lamda = lamda                                                                              # scaling parameter for SUTKF and SLUTKF
        
        if rs == None:
            rs = np.random.RandomState()

        self.rs = rs                                                                                    # random number generator

        # initial state
        if self.enkf_method == 4:                                                                       # UTKF
            self.x = np.zeros((self.nem, self.N), float)
            P = self.proc_q*np.eye(self.N)
            sigma_mtx = np.linalg.cholesky(float(self.N)*P).T                                           # upper triangular matrix obtained by Cholesky decomposition
            if self.sp_dist == 1:                                                                       # normal distribution of ensemble when self.sp_dist = 1
                if self.sp_dist_nrml == 0:                                                              # sp_dist_nrml == 0: each element
                    rn_list = np.zeros((self.N, self.nem_sclfct - 1), float)
                elif self.sp_dist_nrml == 1:                                                            # sp_dist_nrml == 1: entire element
                    rn_list = np.zeros(self.nem_sclfct - 1, float)
                    n_rng = get_truncated_normal(0., 1., 0., 1.)                                        # mean, spread, a, b; standard normal truncated to the range (a, b)
                    rn_list[:] = n_rng.rvs(self.nem_sclfct - 1)                    

            for n in range(self.N):
                if self.sp_dist == 0:                                                                   # uniform distribution of ensemble
                    for i in range(self.nem_sclfct):
                        self.x[(2*self.nem_sclfct*n)+i,:] = F + (i+1)/self.nem_sclfct*sigma_mtx[:,n]
                        self.x[(2*self.nem_sclfct*n)+self.nem_sclfct+i,:] = F - (i+1)/self.nem_sclfct*sigma_mtx[:,n]
                elif self.sp_dist == 1:                                                                 # normal distribution of ensemble
                    if self.sp_dist_nrml == 0:                                                          # sp_dist_nrml == 0: each element
                        if self.nem_sclfct > 1:
                            for j in range(self.N):
                                if sigma_mtx[j,n] == 0.:
                                    rn_list[j,:] = 0.
                                else:
                                    n_rng = get_truncated_normal(0, np.abs(np.sqrt(P[j,n])), 0, np.abs(sigma_mtx[j,n]))         # mean, spread, a, b; standard normal truncated to the range (a, b)
                                    if sigma_mtx[j,n] < 0.:
                                        rn_list[j,:] = -n_rng.rvs(self.nem_sclfct - 1)
                                    else:
                                        rn_list[j,:] = n_rng.rvs(self.nem_sclfct - 1)
                                
                        for i in range(self.nem_sclfct):
                            if i == np.arange(self.nem_sclfct).max():
                                self.x[(2*self.nem_sclfct*n)+i,:] = F + sigma_mtx[:,n]
                                self.x[(2*self.nem_sclfct*n)+self.nem_sclfct+i,:] = F - sigma_mtx[:,n]
                            else:
                                self.x[(2*self.nem_sclfct*n)+i,:] = F + rn_list[:,i]
                                self.x[(2*self.nem_sclfct*n)+self.nem_sclfct+i,:] = F - rn_list[:,i]
                    elif self.sp_dist_nrml == 1:                                                        # sp_dist_nrml == 1: entire element
                        for i in range(self.nem_sclfct):
                            if i == np.arange(self.nem_sclfct).max():
                                self.x[(2*self.nem_sclfct*n)+i,:] = F + sigma_mtx[:,n]
                                self.x[(2*self.nem_sclfct*n)+self.nem_sclfct+i,:] = F - sigma_mtx[:,n]
                            else:
                                self.x[(2*self.nem_sclfct*n)+i,:] = F + rn_list[i] * sigma_mtx[:,n]
                                self.x[(2*self.nem_sclfct*n)+self.nem_sclfct+i,:] = F - rn_list[i] * sigma_mtx[:,n] 
        elif self.enkf_method == 5:                                                                     # LUTKF
            self.x = np.zeros((self.nem, self.N), float)
            sigma = np.sqrt(float(self.ndim_loc)*self.proc_q)

            if self.sp_dist == 0:                                                                       # uniform distribution of ensemble
                for i in range(self.nem_sclfct):     
                    self.x[i,:] = F + (i+1)/self.nem_sclfct*sigma
                    self.x[self.nem_sclfct+i,:] = F - (i+1)/self.nem_sclfct*sigma
            elif self.sp_dist == 1:                                                                     # normal distribution of ensemble
                if self.nem_sclfct > 1:
                    n_rng = get_truncated_normal(0, np.sqrt(self.proc_q), 0, sigma)                     # mean, spread, a, b; standard normal truncated to the range (a, b) 
                    rn_list = n_rng.rvs(self.nem_sclfct - 1)
                    
                for i in range(self.nem_sclfct):
                    if i == np.arange(self.nem_sclfct).max():
                        self.x[i,:] = F + sigma
                        self.x[self.nem_sclfct+i,:] = F - sigma
                    else:
                        self.x[i,:] = F + rn_list[i]
                        self.x[self.nem_sclfct+i,:] = F - rn_list[i]        
        elif self.enkf_method == 6:                                                                     # SUTKF
            self.x = np.zeros((self.nem, self.N), float)
            P = self.proc_q*np.eye(self.N)
            sigma_mtx = np.linalg.cholesky((float(self.en) + self.lamda)*P).T                           # upper triangular matrix obtained by Cholesky decomposition
            if self.sp_dist == 1:
                if self.sp_dist_nrml == 0:                                                              # normal distribution of ensemble when self.sp_dist = 1 (0: each element)
                    rn_list = np.zeros((self.N, self.nem_sclfct - 1), float)
                elif self.sp_dist_nrml == 1:                                                            # normal distribution of ensemble when self.sp_dist = 1 (1: entire element)
                    rn_list = np.zeros(self.nem_sclfct - 1, float)
                    n_rng = get_truncated_normal(0., 1., 0., 1.)                                        # mean, spread, a, b; standard normal truncated to the range (a, b)
                    rn_list[:] = n_rng.rvs(self.nem_sclfct - 1)

            for n in range(self.N):
                if self.sp_dist == 0:                                                                   # uniform distribution of ensemble
                    self.x[0,:] = F
                    
                    for i in range(self.nem_sclfct):
                        self.x[1+(2*self.nem_sclfct*n)+i,:] = F + (i+1)/self.nem_sclfct*sigma_mtx[:,n]
                        self.x[1+(2*self.nem_sclfct*n)+self.nem_sclfct+i,:] = F - (i+1)/self.nem_sclfct*sigma_mtx[:,n]
                elif self.sp_dist == 1:                                                                 # normal distribution of ensemble
                    if self.sp_dist_nrml == 0:                                                                             
                        if self.nem_sclfct > 1:
                            for j in range(self.N):
                                if sigma_mtx[j,n] == 0.:
                                    rn_list[j,:] = 0.
                                else:
                                    n_rng = get_truncated_normal(0, np.abs(np.sqrt(P[j,n])), 0, np.abs(sigma_mtx[j,n]))         # mean, spread, a, b; standard normal truncated to the range (a, b)
                                    if sigma_mtx[j,n] < 0.:
                                        rn_list[j,:] = -n_rng.rvs(self.nem_sclfct - 1)
                                    else:
                                        rn_list[j,:] = n_rng.rvs(self.nem_sclfct - 1)

                        self.x[0,:] = F
                                
                        for i in range(self.nem_sclfct):
                            if i == np.arange(self.nem_sclfct).max():
                                self.x[1+(2*self.nem_sclfct*n)+i,:] = F + sigma_mtx[:,n]
                                self.x[1+(2*self.nem_sclfct*n)+self.nem_sclfct+i,:] = F - sigma_mtx[:,n]
                            else:
                                self.x[1+(2*self.nem_sclfct*n)+i,:] = F + rn_list[:,i]
                                self.x[1+(2*self.nem_sclfct*n)+self.nem_sclfct+i,:] = F - rn_list[:,i]
                    elif self.sp_dist_nrml == 1:
                        self.x[0,:] = F
                        
                        for i in range(self.nem_sclfct):
                            if i == np.arange(self.nem_sclfct).max():
                                self.x[1+(2*self.nem_sclfct*n)+i,:] = F + sigma_mtx[:,n]
                                self.x[1+(2*self.nem_sclfct*n)+self.nem_sclfct+i,:] = F - sigma_mtx[:,n]
                            else:
                                self.x[1+(2*self.nem_sclfct*n)+i,:] = F + rn_list[i] * sigma_mtx[:,n]
                                self.x[1+(2*self.nem_sclfct*n)+self.nem_sclfct+i,:] = F - rn_list[i] * sigma_mtx[:,n] 
        elif self.enkf_method == 7:                                                                     # SLUTKF
            self.x = np.zeros((self.nem, self.N), float)
            sigma = np.sqrt((float(self.en) + self.lamda)*self.proc_q)

            if self.sp_dist == 0:                                                                       # uniform distribution of ensemble
                self.x[0,:] = F
                
                for i in range(self.nem_sclfct):     
                    self.x[1+i,:] = F + (i+1)/self.nem_sclfct*sigma
                    self.x[1+self.nem_sclfct+i,:] = F - (i+1)/self.nem_sclfct*sigma
            elif self.sp_dist == 1:                                                                     # normal distribution of ensemble
                if self.nem_sclfct > 1:
                    n_rng = get_truncated_normal(0, np.sqrt(self.proc_q), 0, sigma)                     # mean, spread, a, b; standard normal truncated to the range (a, b) 
                    rn_list = n_rng.rvs(self.nem_sclfct - 1)

                self.x[0,:] = F
                
                for i in range(self.nem_sclfct):
                    if i == np.arange(self.nem_sclfct).max():
                        self.x[1+i,:] = F + sigma
                        self.x[1+self.nem_sclfct+i,:] = F - sigma
                    else:
                        self.x[1+i,:] = F + rn_list[i]
                        self.x[1+self.nem_sclfct+i,:] = F - rn_list[i]
            elif self.sp_dist == 2:                                                                     # normal distribution of ensemble using box muller transform (nrml_bmt)
                self.x[0,:] = F
                
                for i in range(self.nem_sclfct):     
                    self.x[1+i,:] = F + self.nrml_bmt[i]*sigma
                    self.x[1+self.nem_sclfct+i,:] = F - self.nrml_bmt[i]*sigma                        
        else:
            self.x = F + np.sqrt(self.proc_q)*self.rs.standard_normal(size=(self.nem, self.N))
            
        # state wrapping for parallel operation
        self.xwrap = np.zeros((self.nem, self.N+3), float)                                              # N+3 for x_i-2, x_i-1, x_i+1 

        # initial external forcing
        if self.deltaF == 0.:
            self.forcing = self.F
        else:
            self.forcing = self.rs.gamma(self.F/self.deltaF, self.deltaF, size=(self.nem, self.N))      # [optional] forcing using gamma distribution 

    def shiftx(self):
        xwrap = self.xwrap
        xwrap[:,2:self.N+2] = self.x 
        xwrap[:,1] = self.x[:,-1]
        xwrap[:,0] = self.x[:,-2]
        xwrap[:,-1] = self.x[:,0]
        xm2 = xwrap[:,0:self.N]                                                                         # x_39 x_40 x_1  ... x_37 x_38  for N = 40
        xm1 = xwrap[:,1:self.N+1]                                                                       # x_40 x_1  x_2  ... x_38 x_39  for N = 40
        xp1 = xwrap[:,3:self.N+3]                                                                       # x_2  x_3  x_4  ... x_40 x_1   for N = 40
        
        return xm2,xm1,xp1

    def dxdt(self):
        xm2,xm1,xp1 = self.shiftx()                                                                     # for parallel operation

        return (xp1 - xm2)*xm1 - self.blend*self.x + self.forcing     

    def fcst(self, prior_additive_noise=False):
        # Due to gamma distribution with a shape parameter self.F/deltaF and a scale parameter deltaF: 1) mean = self.F/deltaF * deltaF; 2) variance = self.F/deltaF * deltaF ** 2; 3) std = sqrt(variance), 
        # gamma distributed forcing with mean = self.F. Approaches constant value self.F as deltaF approaches zero, more variability as deltaF increases. 
        # Forcing is random at every grid point, but correlated in time with lag-1 (i.e., previous) correlation of self.Fcorr.  
        # Forcing is assumed constant over one time step (does not vary inside RK4).
        if self.deltaF == 0.:
            self.forcing = self.F
        else:
            # adjust deltaF to preserve expected variance
            deltaF = self.deltaF/(1. - np.sqrt(self.Fcorr))
            self.forcing = self.Fcorr*self.forcing + (1. - self.Fcorr)*self.rs.gamma(self.F/deltaF, deltaF, size=(self.nem, self.N))     # forcing using gamma distribution [optional]
           
        # Integration using fourth-order Runge-Kutta scheme
        # Given that dxdt = f(x(t)),
        # k1 = f(x(t))*h
        # k2 = f(x(t)+k1/2)*h
        # k3 = f(x(t)+k2/2)*h
        # k4 = f(x(t)+k3)*h
        # x(t+1) = x(t)+(k1+2(k2+k3)+k4)/6
        # for simplicity
        # k1 = f(x(t))
        # k2 = f(x(t)+k1*h/2)
        # k3 = f(x(t)+k2*h/2)
        # k4 = f(x(t)+k3*h)
        # x(t+1) = x(t)+(k1+2(k2+k3)+k4)*h/6
        h = self.dt
        h2 = h/2.
        h6 = h/6.
        
        x = self.x
        dxdt1 = self.dxdt()
        self.x = x + dxdt1*h2
        dxdt2 = self.dxdt()
        self.x = x + dxdt2*h2
        dxdt3 = self.dxdt()
        self.x = x + dxdt3*h
        dxdt4 = self.dxdt()
        self.x = x + (dxdt1 + dxdt4 + 2.0*(dxdt2 + dxdt3))*h6

        if prior_additive_noise:                                                                        # in case of PF and LPF
            self.x = self.x + np.sqrt(self.proc_q)*self.rs.standard_normal(size=(self.nem, self.N))
