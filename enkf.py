import numpy as np
import sys
#from numpy.linalg import eigh # scipy.linalg.eigh broken on my mac
from scipy.linalg import eigh, cho_solve, cho_factor, svd, pinvh, solve_triangular, cholesky
import scipy.stats as stats

def ks_test_normal_dist(x, alpha = 0.05):
    """
    stats.kstest returns:

              1. KS statistic
              2. pvalue

    if pvalue > 0.05 (5%) we accept the null hypothesis:

    H0: our random variable from simulation follow the distribution with
        the parameters obtained from 'stats.dist.fit'. 
    """
    ks = stats.kstest(x, 'norm', stats.norm.fit(x))
    p_value = ks[1]
    result = 'accept' if p_value > alpha else 'reject'
    
    return (p_value, result)    
    
def ks_test_white_noise(x, alpha = 0.05):
    """ Kolmogorov-Smirnov test """
    ks = stats.kstest(x, 'norm', (0.0, np.std(x)))
    p_value = ks[1]
    result = 'accept' if p_value > alpha else 'reject'
    
    return (p_value, result)
    
def nonlinear_h(x, h_type):
    """ observation operator H """
    if h_type == 0:                 # linear h(x)=x
        return x
    elif h_type == 1:               # nonlinear h(x)=|x|
        return np.abs(x)
    elif h_type == 2:               # nonlinear h(x)=ln(|x|) 
        return np.log(np.abs(x))

def nonlinear_h_deriv(y, h_type, eps=1.e-8):
    """Derivative of nonlinear_h(y, h_type)."""
    if h_type == 0:
        return np.ones_like(y)
    elif h_type == 1:
        return np.sign(y)
    elif h_type == 2:
        y_safe = np.where(np.abs(y) < eps, eps*np.sign(y + eps), y)
        return 1.0 / y_safe

def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    stats.random_state = np.random.RandomState(seed=20)
    return stats.truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)
        
def symsqrt_psd(a, inv=False):
    """symmetric square-root of a symmetric positive definite matrix"""
    evals, eigs = eigh(a)
    symsqrt =  (eigs * np.sqrt(np.maximum(evals,0))).dot(eigs.T)
    if inv:
        inv =  (eigs * (1./np.maximum(evals,0))).dot(eigs.T)
        return symsqrt, inv
    else:
        return symsqrt
        
def linalg_solve(a,b,method):
    """determine x in linear equation ax=b"""
    if method == 1:
        return np.linalg.solve(a,b) 
    elif method == 2:
        f = cho_factor(a)
        return cho_solve(f,b)    

def symsqrtinv_psd(a):
    """inverse and inverse symmetric square-root of a symmetric positive definite matrix"""
    try:
        evals, evecs = eigh(a)
        symsqrtinv =  (evecs * (1./np.sqrt(np.maximum(evals,0)))).dot(evecs.T)
        inv =  (evecs * (1./np.maximum(evals,0))).dot(evecs.T)
    except:
        symsqrtinv = 0.
        inv = 0.
        
    return symsqrtinv, inv

def syminv_psd(a):
    """inverse of a symmetric positive definite matrix"""
    try:
        evals, evecs = eigh(a)
        inv =  (evecs * (1./np.maximum(evals,0))).dot(evecs.T)
    except:
        inv = 0.
        
    return inv
    

def serial_ensrf(xmean, xptb, h, h_type, obs, obs_r, obs_rloc):
    nem, ndim = xptb.shape
    nobs = obs.shape[-1]

    for nob, ob in zip(np.arange(nobs), obs):
        hxmean_linear = np.dot(h[nob], xmean)
        hxmean = nonlinear_h(hxmean_linear, h_type)
        
        dh = nonlinear_h_deriv(np.array(hxmean_linear), h_type)
    
        h_eff = dh * h[nob]
        zptb = np.dot(xptb, h_eff)

        hxens = zptb.reshape((nem, 1))
        D = (hxens[:, 0] ** 2).sum() / float(nem - 1) + obs_r

        gainfact = np.sqrt(D) / (np.sqrt(D) + np.sqrt(obs_r))
        pbht = (xptb.T * hxens[:, 0]).sum(axis=1) / float(nem - 1)

        kfgain = obs_rloc.T[nob, :] * pbht / D
        xmean = xmean + kfgain * (ob - hxmean)

        xptb = xptb - gainfact * kfgain * hxens

    return xmean, xptb, 1.


"""
def serial_ensrf(xmean,xptb,h,obs,obs_r,obs_rloc):
    # serial potter method [when h=0 (linear operator)]      # serial: observation are processed one at a time
    nem, ndim = xptb.shape
    nobs = obs.shape[-1]

    for nob,ob in zip(np.arange(nobs),obs):
        # forward operator.
        zptb = np.dot(xptb,h[nob])  # h: linear operator
        hxmean = np.dot(h[nob],xmean)
        # state space update
        hxens = zptb.reshape((nem, 1))
        D = (hxens**2).sum()/(nem-1) + obs_r
        gainfact = np.sqrt(D)/(np.sqrt(D)+np.sqrt(obs_r))
        pbht = (xptb.T*hxens[:,0]).sum(axis=1)/float(nem-1)
        kfgain = obs_rloc[nob,:]*pbht/D        # B localization (when h=0, ith observation has an great impact on ith model variable.)
        xmean = xmean + kfgain*(ob-hxmean)
        xptb = xptb - gainfact*kfgain*hxens
    return xmean, xptb, 1.
"""

def serial_ensrf_modens(xmean,xptb,h,obs,obs_r,obs_rloc,covlocal,z):
    """serial potter method [when h=0 (linear operator)]"""      # serial: observation are processed one at a time
    nem, ndim = xptb.shape
    nobs = obs.shape[-1]

    # if True, use gain from modulated ensemble to
    # update perts.  if False, use gain from original ensemble.
    update_xprime = True
    if z is None:
        # set ensemble to square root of localized Pb
        Pb = covlocal*np.dot(xptb.T,xptb)/(nem-1)
        evals, eigs = eigh(Pb)
        evals = np.where(evals > 1.e-10, evals, 1.e-10)
        nanals2 = eigs.shape[0]
        xprime2 = np.sqrt(nanals2-1)*(eigs*np.sqrt(evals)).T
    else:
        # modulation ensemble
        neig = z.shape[0]; nanals2 = neig*nem; nanal2 = 0
        xprime2 = np.zeros((nanals2,ndim),xptb.dtype)
        for j in range(neig):
            for nanal in range(nem):
                xprime2[nanal2,:] = xptb[nanal,:]*z[neig-j-1,:]
                # unmodulated member is j=1, scaled by z[-1]
                nanal2 += 1
        xprime2 = np.sqrt(float(nanals2-1)/float(nem-1))*xprime2

    # update xmean using full xprime2
    # update original xptb using gain from full xprime2
    for nob,ob in zip(np.arange(nobs),obs):
        # forward operator.
        zptb = np.dot(xprime2,h[nob])   # h: linear operator
        hxprime_orig = np.dot(xptb,h[nob])
        hxmean = np.dot(h[nob],xmean)
        # state space update
        hxens = zptb.reshape((nanals2, 1))
        hxens_orig = hxprime_orig.reshape((nem, 1))
        D = (hxens**2).sum()/(nanals2-1) + obs_r
        gainfact = np.sqrt(D)/(np.sqrt(D)+np.sqrt(obs_r))
        pbht = (xprime2.T*hxens[:,0]).sum(axis=1)/float(nanals2-1)
        kfgain = pbht/D
        xmean = xmean + kfgain*(ob-hxmean)
        xprime2 = xprime2 - gainfact*kfgain*hxens
        if not update_xprime:
            D = (hxens_orig**2).sum()/(nem-1) + obs_r
            gainfact = np.sqrt(D)/(np.sqrt(D)+np.sqrt(obs_r))
            pbht = (xptb.T*hxens_orig[:,0]).sum(axis=1)/float(nem-1)
            kfgain = obs_rloc[nob,:]*pbht/D
        xptb  = xptb  - gainfact*kfgain*hxens_orig
    return xmean, xptb, 1.





def bulk_ensrf(xmean, xptb, h, h_type, obs, obs_r, covlocal, denkf=False):
    
    #Bulk EnSRF / Potter method with original model-space B localization.

    #  Original B localization: Pb_loc = covlocal * Pb
    #  Nonlinear observations are treated using a first-order linearized H
    #  around the background ensemble mean.
    
    nem, ndim = xptb.shape
    nobs = obs.shape[-1]

    R = obs_r * np.eye(nobs)
    Rsqrt = np.sqrt(obs_r) * np.eye(nobs)

    # 1) Original B-localization
    Pb = np.dot(xptb.T, xptb) / float(nem - 1)
    Pb = covlocal * Pb

    # 2) Nonlinear observation mean
    hxmean_linear = np.dot(h, xmean)
    hxmean = nonlinear_h(hxmean_linear, h_type)

    # 3) Linearized observation operator around xmean
    dh = nonlinear_h_deriv(hxmean_linear, h_type)
    h_eff = dh[:, None] * h

    # 4) Bulk EnSRF update
    D = np.dot(np.dot(h_eff, Pb), h_eff.T) + R

    if not denkf:
        Dsqrt, Dinv = symsqrt_psd(D, inv=True)
    else:
        Dinv = cho_solve(cho_factor(D), np.eye(nobs))

    kfgain = np.dot(np.dot(Pb, h_eff.T), Dinv)

    if not denkf:
        tmp = Dsqrt + Rsqrt
        tmpinv = cho_solve(cho_factor(tmp), np.eye(nobs))
        gainfact = np.dot(Dsqrt, tmpinv)
        reducedgain = np.dot(kfgain, gainfact)
    else:
        reducedgain = 0.5 * kfgain

    # 5) Mean update
    xmean = xmean + np.dot(kfgain, obs - hxmean)

    # 6) Perturbation update
    zptb = np.empty((nem, nobs), xptb.dtype)
    for nanal in range(nem):
        zptb[nanal] = np.dot(h_eff, xptb[nanal])

    xptb = xptb - np.dot(reducedgain, zptb.T).T

    return xmean, xptb, 1.

"""
def bulk_ensrf(xmean,xptb,h,obs,obs_r,covlocal,denkf=False):
    # bulk potter method [when h=0 (linear operator)]                   # bulk: observations are processed in batches
    nem, ndim = xptb.shape
    nobs = obs.shape[-1]

    R = obs_r*np.eye(nobs)
    Rsqrt = np.sqrt(obs_r)*np.eye(nobs)
    Pb = np.dot(np.transpose(xptb),xptb)/(nem-1)
    Pb = covlocal*Pb                                                        # B localization 
    D = np.dot(np.dot(h,Pb),h.T)+R                                          # h: linear operator
    if not denkf:
        Dsqrt,Dinv = symsqrt_psd(D,inv=True)
    else:
        Dinv = cho_solve(cho_factor(D),np.eye(nobs))                        # determine X by sloving DX = I
    kfgain = np.dot(np.dot(Pb,h.T),Dinv)
    if not denkf:
        tmp = Dsqrt + Rsqrt
        tmpinv = cho_solve(cho_factor(tmp),np.eye(nobs))                    # determine X by sloving tmp*X = I
        gainfact = np.dot(Dsqrt,tmpinv)
        reducedgain = np.dot(kfgain, gainfact)
    else:
        reducedgain = 0.5*kfgain
    xmean = xmean + np.dot(kfgain, obs-np.dot(h,xmean))
    zptb = np.empty((nem, nobs), xptb.dtype)
    for nanal in range(nem):
        zptb[nanal] = np.dot(h,xptb[nanal])
    xptb = xptb - np.dot(reducedgain,zptb.T).T
    return xmean, xptb, 1.
"""

def enkf(xmean,xptb,h,obs,obs_r,covlocal,rs):
    """bulk enkf method with perturbed obs [when h=0 (linear operator)]"""  # bulk: observations are processed in batches
    nem, ndim = xptb.shape
    nobs = obs.shape[-1]

    R = obs_r*np.eye(nobs)
    #Rsqrt = np.sqrt(obs_r)*np.eye(nobs)
    Pb = np.dot(np.transpose(xptb),xptb)/(nem-1)
    Pb = covlocal*Pb                                                        # B localization
    D = np.dot(np.dot(h,Pb),h.T)+R                                          # h: linear operator
    Dinv = cho_solve(cho_factor(D),np.eye(nobs))                            # determine X by sloving DX = I
    kfgain = np.dot(np.dot(Pb,h.T),Dinv)
    xmean = xmean + np.dot(kfgain, obs-np.dot(h,xmean))
    zptb = np.empty((nem, nobs), xptb.dtype)
    for nanal in range(nem):
        zptb[nanal] = np.dot(h,xptb[nanal])
    xptb = xptb - np.dot(kfgain, zptb.T).T
    return xmean, xptb, 1.

def ptbobs_enkf(xmean,xptb,h,obs,obs_r,covlocal,rs):
    """bulk enkf method with perturbed obs [when h=0 (linear operator)]"""   # bulk: observations are processed in batches
    nem, ndim = xptb.shape
    nobs = obs.shape[-1]

    R = obs_r*np.eye(nobs)
    #Rsqrt = np.sqrt(obs_r)*np.eye(nobs)
    Pb = np.dot(np.transpose(xptb),xptb)/(nem-1)
    Pb = covlocal*Pb                                                        # B localization
    D = np.dot(np.dot(h,Pb),h.T)+R                                          # h: linear operator
    Dinv = cho_solve(cho_factor(D),np.eye(nobs))                            # determine X by sloving DX = I
    kfgain = np.dot(np.dot(Pb,h.T),Dinv)
    xmean = xmean + np.dot(kfgain, obs-np.dot(h,xmean))
    obnoise = np.sqrt(obs_r)*rs.standard_normal(size=(nem,nobs))
    obnoise_var = ((obnoise-obnoise.mean(axis=0))**2).sum(axis=0)/(nem-1)
    obnoise = np.sqrt(obs_r)*obnoise/np.sqrt(obnoise_var)
    zptb = np.empty((nem, nobs), xptb.dtype)
    for nanal in range(nem):
        zptb[nanal] = np.dot(h,xptb[nanal]) - obnoise[nanal]
    xptb = xptb - np.dot(kfgain, zptb.T).T
    return xmean, xptb, 1.

def etkf(xmean, xptb, h, h_type, obs, obs_r):
    """ETKF (use only with full rank ensemble, no localization)"""
    nem, ndim = xptb.shape                                                      # xptb: nems x ndim
    nobs = obs.shape[-1]                                                        # obs: 1 x nobs(ndim)

    zptb = np.zeros((nem, nobs), xptb.dtype)

    """
    for iens in range(nem):
        zptb[iens] = np.dot(h,xptb[iens])
        zptb[iens] = nonlinear_h(zptb[iens], h_type)

    zmean = np.dot(h,xmean)
    zmean = nonlinear_h(zmean, h_type)
    zrsd = obs - zmean
    """

    x = xmean + xptb
    z = np.zeros((nem, nobs), x.dtype)
    
    for iens in range(nem):
        z[iens] = np.dot(h, x[iens])
        z[iens] = nonlinear_h(z[iens], h_type)
   
    zmean = z.mean(axis=0)
    zptb = z - zmean
    zrsd = obs - zmean    

    # calculate analysis mean and perturbation of all grids (each grid has one state variable) for global data assimilation 
    Rinv = (1./obs_r)*np.eye(nobs)
    C = np.dot(zptb,Rinv)                                                       # zptb: (Yb)^T
    sqrt_pa, pa = symsqrtinv_psd((nem-1)*np.eye(nem)+np.dot(C,zptb.T))          # symmetric matrix inverse and its square root using eigen decomposition
    K = np.dot(xptb.T,np.dot(pa,C))
    Wa = np.sqrt(nem-1)*sqrt_pa
    xmean = xmean + np.dot(K, zrsd)
    xptb = np.dot(Wa.T,xptb)
    beta_mf = 1.                                                                # weight for Gaussian sum filter
    
    return xmean, xptb, zrsd, beta_mf

def utkf(x, xmean, xptb, proc_q, proc_q_covinfl_par, h, h_type, obs, obs_r, nem_sclfct, sp_dist, sp_dist_nrml, nrml_bmt, K_solver, mpi, comm, myrank):
    """UTKF (use only with full rank ensemble, no localization)"""
    if mpi:
        comm.Bcast(x, root=0) 
        comm.Bcast(xmean, root=0)    
        comm.Bcast(xptb, root=0)
        comm.Bcast(obs, root=0)
        
    nem, ndim = x.shape                                                         # x: nem x ndim
    nobs = obs.shape[-1]                                                        # obs: 1 x nobs(ndim)

    w = 1./(2.*float(ndim))
    
    Q = proc_q*proc_q_covinfl_par*np.eye(ndim)
    xptb_w = np.sqrt(w) * xptb
    P = np.dot(xptb_w.T, xptb_w) + Q 

    P_b = P.copy()
    
    z = np.zeros((nem, nobs), x.dtype)
    zptb = np.zeros((nem, nobs), xptb.dtype)
    
    for iem in range(nem):
        z[iem] = np.dot(h, x[iem])
        z[iem] = nonlinear_h(z[iem], h_type)
            
    zmean = z.mean(axis=0)
    zptb = z - zmean
    zptb_w = np.sqrt(w) * zptb
    zrsd = obs - zmean
    
    # calculate analysis mean and perturbation of each grid (each grid has one state variable) for global data assimilation
    if K_solver in [0,1,2]:
        R = obs_r*np.eye(nobs)
        S = np.dot(zptb_w.T, zptb_w) + R
        Pxz = np.dot(xptb_w.T, zptb_w)      
    if K_solver == 0:                                                           # 0: linear solver using matrix inverse (slow)
        Sinv = np.linalg.inv(S)
        beta_mf = np.exp(-0.5*np.dot(np.dot(zrsd.T, Sinv), zrsd))               # weight for Gaussian sum filter
        K = np.dot(Pxz, Sinv)                                                   # K = PxzS^(-1)
        xmean = xmean + np.dot(K, zrsd) 
        P = P - np.dot(Pxz, K.T)
    elif K_solver in [1,2]:                                                     # 1: standard linear solver (fast, safe), 2: linear solver using cholesky factorization(fastest, unsafe)
        beta_mf = 1.                                                            # weight for Gaussian sum filter
        K_t = linalg_solve(S, Pxz.T, K_solver)                                  # SK^t=Pxz^t
        xmean = xmean + np.dot(K_t.T, zrsd)
        P = P - np.dot(Pxz, K_t)
    elif K_solver == 3:                                                     # 3: computation in the subspace spanned by the ensemble (fastest, safe)
        beta_mf = 1. 
        Rinv = (1/obs_r)*np.eye(nobs)
        C = np.dot(zptb_w, Rinv)                                            # zptb: (Yb)^T
        pa = syminv_psd(np.eye(nem)+np.dot(C,zptb_w.T))                        # symmetric matrix inverse and its square root using eigen decomposition
        K = np.dot(xptb_w.T, np.dot(pa,C))
        xmean = xmean + np.dot(K, zrsd)
        P = np.dot(xptb_w.T, np.dot(pa,xptb_w)) + Q 
    
    # sigma point selection
    sigma_mtx = np.linalg.cholesky(float(ndim)*P).T                             # upper triangular matrix obtained by Cholesky decomposition
    if sp_dist == 1:
        if sp_dist_nrml == 0:                                                   # normal distribution of ensemble when self.sp_dist = 1 (0: each element)
            rn_list = np.zeros((ndim, nem_sclfct - 1), float)
        elif sp_dist_nrml == 1:                                                 # normal distribution of ensemble when self.sp_dist = 1 (1: entire element)
            rn_list = np.zeros(nem_sclfct - 1, float)
            n_rng = get_truncated_normal(0., 1., 0., 1.)                        # mean, spread, a, b; standard normal truncated to the range (a, b)
            rn_list[:] = n_rng.rvs(nem_sclfct - 1)
            
    if mpi:
        n = myrank                                                      
        if n < ndim:
            if sp_dist == 0:
                for i in range(nem_sclfct):
                    x[(2*nem_sclfct*n)+i,:] = xmean + (i+1)/nem_sclfct*sigma_mtx[:,n]
                    x[(2*nem_sclfct*n)+nem_sclfct+i,:] = xmean - (i+1)/nem_sclfct*sigma_mtx[:,n]
            elif sp_dist == 1:
                if sp_dist_nrml == 0: 
                    if nem_sclfct > 1:
                        for j in range(ndim):
                            if sigma_mtx[j,n] == 0.:
                                rn_list[j,:] = 0.
                            else:
                                n_rng = get_truncated_normal(0, np.sqrt(np.abs(P[j,n])), 0, np.abs(sigma_mtx[j,n]))        # mean, spread, a, b; standard normal truncated to the range (a, b)
                                if sigma_mtx[j,n] < 0.:
                                    rn_list[j,:] = -n_rng.rvs(nem_sclfct - 1)
                                else:
                                    rn_list[j,:] = n_rng.rvs(nem_sclfct - 1)
                                
                            
                    for i in range(nem_sclfct):
                        if i == np.arange(nem_sclfct).max():
                            x[(2*nem_sclfct*n)+i,:] = xmean + sigma_mtx[:,n]
                            x[(2*nem_sclfct*n)+nem_sclfct+i,:] = xmean - sigma_mtx[:,n]
                        else:
                            x[(2*nem_sclfct*n)+i,:] = xmean + rn_list[:,i]
                            x[(2*nem_sclfct*n)+nem_sclfct+i,:] = xmean - rn_list[:,i]
                elif sp_dist_nrml == 1:
                    for i in range(nem_sclfct):
                        if i == np.arange(nem_sclfct).max():
                            x[(2*nem_sclfct*n)+i,:] = xmean + sigma_mtx[:,n]
                            x[(2*nem_sclfct*n)+nem_sclfct+i,:] = xmean - sigma_mtx[:,n]
                        else:
                            x[(2*nem_sclfct*n)+i,:] = xmean + rn_list[i] * sigma_mtx[:,n]
                            x[(2*nem_sclfct*n)+nem_sclfct+i,:] = xmean - rn_list[i] * sigma_mtx[:,n]
            elif sp_dist == 2:
                for i in range(nem_sclfct):
                    x[(2*nem_sclfct*n)+i,:] = xmean + nrml_bmt[i]*sigma_mtx[:,n]
                    x[(2*nem_sclfct*n)+nem_sclfct+i,:] = xmean - nrml_bmt[i]*sigma_mtx[:,n]  

            comm.Allgather(np.array(x[(2*nem_sclfct*n):(2*nem_sclfct*(n+1)),:]), x)
    else:
        for n in range(ndim):
            if sp_dist == 0:
                for i in range(nem_sclfct):
                    x[(2*nem_sclfct*n)+i,:] = xmean + (i+1)/nem_sclfct*sigma_mtx[:,n]
                    x[(2*nem_sclfct*n)+nem_sclfct+i,:] = xmean - (i+1)/nem_sclfct*sigma_mtx[:,n]
            elif sp_dist == 1:
                if sp_dist_nrml == 0: 
                    if nem_sclfct > 1:
                        for j in range(ndim):
                            if sigma_mtx[j,n] == 0.:
                                rn_list[j,:] = 0.
                            else:
                                n_rng = get_truncated_normal(0, np.sqrt(np.abs(P[j,n])), 0, np.abs(sigma_mtx[j,n]))        # mean, spread, a, b; standard normal truncated to the range (a, b)
                                if sigma_mtx[j,n] < 0.:
                                    rn_list[j,:] = -n_rng.rvs(nem_sclfct - 1)
                                else:
                                    rn_list[j,:] = n_rng.rvs(nem_sclfct - 1)
                                                            
                    for i in range(nem_sclfct):
                        if i == np.arange(nem_sclfct).max():
                            x[(2*nem_sclfct*n)+i,:] = xmean + sigma_mtx[:,n]
                            x[(2*nem_sclfct*n)+nem_sclfct+i,:] = xmean - sigma_mtx[:,n]
                        else:
                            x[(2*nem_sclfct*n)+i,:] = xmean + rn_list[:,i]
                            x[(2*nem_sclfct*n)+nem_sclfct+i,:] = xmean - rn_list[:,i]
                elif sp_dist_nrml == 1:
                    for i in range(nem_sclfct):
                        if i == np.arange(nem_sclfct).max():
                            x[(2*nem_sclfct*n)+i,:] = xmean + sigma_mtx[:,n]
                            x[(2*nem_sclfct*n)+nem_sclfct+i,:] = xmean - sigma_mtx[:,n]
                        else:
                            x[(2*nem_sclfct*n)+i,:] = xmean + rn_list[i] * sigma_mtx[:,n]
                            x[(2*nem_sclfct*n)+nem_sclfct+i,:] = xmean - rn_list[i] * sigma_mtx[:,n]
            elif sp_dist == 2:
                for i in range(nem_sclfct):
                    x[(2*nem_sclfct*n)+i,:] = xmean + nrml_bmt[i]*sigma_mtx[:,n]
                    x[(2*nem_sclfct*n)+nem_sclfct+i,:] = xmean - nrml_bmt[i]*sigma_mtx[:,n]                        
            
    xptb = x - xmean 
    
    return xmean, xptb, np.diag(P_b), np.diag(P), zrsd, beta_mf 

def sutkf(x, xmean, xptb, proc_q, proc_q_covinfl_par, h, h_type, obs, obs_r, en, alpha, beta, kappa, lamda, nem_sclfct, sp_dist, sp_dist_nrml, nrml_bmt, K_solver, mpi, comm, myrank):
    """Scaled-UTKF (use only with full rank ensemble, no localization)"""
    if mpi:
        comm.Bcast(x, root=0) 
        comm.Bcast(xmean, root=0)    
        comm.Bcast(xptb, root=0)
        comm.Bcast(obs, root=0)
        
    nem, ndim = x.shape                                                         # x: nem x ndim
    nobs = obs.shape[-1]                                                        # obs: 1 x nobs(ndim)
    
    wm = np.empty(nem, float)
    wc = np.empty(nem, float)

    wm[0] = lamda / (float(en) + lamda)
    wc[0] = (lamda / (float(en) + lamda)) + (1. - (alpha**2) + beta)
    wm[1:nem+1] = wc[1:nem+1] = 1. / (2.*(float(en) + lamda))
    
    xmean = np.sum(x.T * wm, axis=1)     
    xptb = x - xmean
    xptb_w = (np.sqrt(wc) * xptb.T).T
    
    Q = proc_q*proc_q_covinfl_par*np.eye(ndim)
    P = np.dot(xptb_w.T, xptb_w) + Q                                              # background error covariance

    P_b = P.copy()
    
    z = np.zeros((nem, nobs), x.dtype)
    zptb = np.zeros((nem, nobs), xptb.dtype)
    
    for iem in range(nem):
        z[iem] = np.dot(h, x[iem])
        z[iem] = nonlinear_h(z[iem], h_type)
            
    zmean = np.sum(z.T * wm, axis=1)    
    zptb = z - zmean
    zptb_w = (np.sqrt(wc) * zptb.T).T
    zrsd = obs - zmean
    
    # calculate analysis mean and perturbation of each grid (each grid has one state variable) for global data assimilation
    if K_solver in [0,1,2]:
        R = obs_r*np.eye(nobs)
        S = np.dot(zptb_w.T, zptb_w) + R
        Pxz = np.dot(xptb_w.T, zptb_w)

    if K_solver == 0:                                                           # 0: linear solver using matrix inverse (slow)
        Sinv = np.linalg.inv(S)
        beta_mf = np.exp(-0.5*np.dot(np.dot(zrsd.T, Sinv), zrsd))               # weight for Gaussian sum filter
        K = np.dot(Pxz, Sinv)                                                   # K = PxzS^(-1)
        xmean = xmean + np.dot(K, zrsd)
        P = P - np.dot(Pxz, K.T)
    elif K_solver in [1,2]:                                                     # 1: standard linear solver (fast, safe), 2: linear solver using cholesky factorization(fastest, unsafe)
        beta_mf = 1.                                                            # weight for Gaussian sum filter
        K_t = linalg_solve(S, Pxz.T, K_solver)                                  # SK^t=Pxz^t
        xmean = xmean + np.dot(K_t.T, zrsd)
        P = P - np.dot(Pxz, K_t)
    elif K_solver == 3:                                                         # 3: computation in the subspace spanned by the ensemble (fastest, safe)
        beta_mf = 1. 
        Rinv = (1/obs_r)*np.eye(nobs)
        C = np.dot(zptb_w, Rinv)                                                # zptb: (Yb)^T
        pa = syminv_psd(np.eye(nem)+np.dot(C,zptb_w.T))                            # symmetric matrix inverse using eigen decomposition
        K = np.dot(xptb_w.T, np.dot(pa,C))
        xmean = xmean + np.dot(K, zrsd)
        P = np.dot(xptb_w.T, np.dot(pa,xptb_w)) + Q
            
    # sigma point selection
    sigma_mtx = np.linalg.cholesky((float(en) + lamda)*P).T                     # upper triangular matrix obtained by Cholesky decomposition
    if sp_dist == 1:
        if sp_dist_nrml == 0:                                                   # normal distribution of ensemble when self.sp_dist = 1 (0: each element)
            rn_list = np.zeros((ndim, nem_sclfct - 1), float)
        elif sp_dist_nrml == 1:                                                 # normal distribution of ensemble when self.sp_dist = 1 (1: entire element)
            rn_list = np.zeros(nem_sclfct - 1, float)
            n_rng = get_truncated_normal(0., 1., 0., 1.)                        # mean, spread, a, b; standard normal truncated to the range (a, b)
            rn_list[:] = n_rng.rvs(nem_sclfct - 1)
            
    if mpi:
        n = myrank                                                      
        if n < ndim:
            if sp_dist == 0:
                x[0,:] = xmean
                
                for i in range(nem_sclfct):
                    x[1+(2*nem_sclfct*n)+i,:] = xmean + (i+1)/nem_sclfct*sigma_mtx[:,n]
                    x[1+(2*nem_sclfct*n)+nem_sclfct+i,:] = xmean - (i+1)/nem_sclfct*sigma_mtx[:,n]
            elif sp_dist == 1:
                if sp_dist_nrml == 0:
                    if nem_sclfct > 1:
                        for j in range(ndim):
                            if sigma_mtx[j,n] == 0.:
                                rn_list[j,:] = 0.
                            else:
                                n_rng = get_truncated_normal(0, np.sqrt(np.abs(P[j,n])), 0, np.abs(sigma_mtx[j,n]))        # mean, spread, a, b; standard normal truncated to the range (a, b)
                                if sigma_mtx[j,n] < 0.:
                                    rn_list[j,:] = -n_rng.rvs(nem_sclfct - 1)
                                else:
                                    rn_list[j,:] = n_rng.rvs(nem_sclfct - 1)
                                
                    x[0,:] = xmean
                    
                    for i in range(nem_sclfct):
                        if i == np.arange(nem_sclfct).max():
                            x[1+(2*nem_sclfct*n)+i,:] = xmean + sigma_mtx[:,n]
                            x[1+(2*nem_sclfct*n)+nem_sclfct+i,:] = xmean - sigma_mtx[:,n]
                        else:
                            x[1+(2*nem_sclfct*n)+i,:] = xmean + rn_list[:,i]
                            x[1+(2*nem_sclfct*n)+nem_sclfct+i,:] = xmean - rn_list[:,i]
                elif sp_dist_nrml == 1:
                    x[0,:] = xmean
                    
                    for i in range(nem_sclfct):
                        if i == np.arange(nem_sclfct).max():
                            x[1+(2*nem_sclfct*n)+i,:] = xmean + sigma_mtx[:,n]
                            x[1+(2*nem_sclfct*n)+nem_sclfct+i,:] = xmean - sigma_mtx[:,n]
                        else:
                            x[1+(2*nem_sclfct*n)+i,:] = xmean + rn_list[i] * sigma_mtx[:,n]
                            x[1+(2*nem_sclfct*n)+nem_sclfct+i,:] = xmean - rn_list[i] * sigma_mtx[:,n]
            elif sp_dist == 2:
                x[0,:] = xmean
                
                for i in range(nem_sclfct):
                    x[1+(2*nem_sclfct*n)+i,:] = xmean + nrml_bmt[i]*sigma_mtx[:,n]
                    x[1+(2*nem_sclfct*n)+nem_sclfct+i,:] = xmean - nrml_bmt[i]*sigma_mtx[:,n]  
                    
            comm.Allgather(np.array(x[1+(2*nem_sclfct*n):1+(2*nem_sclfct*(n+1)),:]), x[1:])
    else:    
        for n in range(ndim):
            if sp_dist == 0:
                x[0,:] = xmean
                
                for i in range(nem_sclfct):
                    x[1+(2*nem_sclfct*n)+i,:] = xmean + (i+1)/nem_sclfct*sigma_mtx[:,n]
                    x[1+(2*nem_sclfct*n)+nem_sclfct+i,:] = xmean - (i+1)/nem_sclfct*sigma_mtx[:,n]
            elif sp_dist == 1:
                if sp_dist_nrml == 0:
                    if nem_sclfct > 1:
                        for j in range(ndim):
                            if sigma_mtx[j,n] == 0.:
                                rn_list[j,:] = 0.
                            else:
                                n_rng = get_truncated_normal(0, np.sqrt(np.abs(P[j,n])), 0, np.abs(sigma_mtx[j,n]))        # mean, spread, a, b; standard normal truncated to the range (a, b)
                                if sigma_mtx[j,n] < 0.:
                                    rn_list[j,:] = -n_rng.rvs(nem_sclfct - 1)
                                else:
                                    rn_list[j,:] = n_rng.rvs(nem_sclfct - 1)
                                
                    x[0,:] = xmean
                    
                    for i in range(nem_sclfct):
                        if i == np.arange(nem_sclfct).max():
                            x[1+(2*nem_sclfct*n)+i,:] = xmean + sigma_mtx[:,n]
                            x[1+(2*nem_sclfct*n)+nem_sclfct+i,:] = xmean - sigma_mtx[:,n]
                        else:
                            x[1+(2*nem_sclfct*n)+i,:] = xmean + rn_list[:,i]
                            x[1+(2*nem_sclfct*n)+nem_sclfct+i,:] = xmean - rn_list[:,i]
                elif sp_dist_nrml == 1:
                    x[0,:] = xmean
                    
                    for i in range(nem_sclfct):
                        if i == np.arange(nem_sclfct).max():
                            x[1+(2*nem_sclfct*n)+i,:] = xmean + sigma_mtx[:,n]
                            x[1+(2*nem_sclfct*n)+nem_sclfct+i,:] = xmean - sigma_mtx[:,n]
                        else:
                            x[1+(2*nem_sclfct*n)+i,:] = xmean + rn_list[i] * sigma_mtx[:,n]
                            x[1+(2*nem_sclfct*n)+nem_sclfct+i,:] = xmean - rn_list[i] * sigma_mtx[:,n]
            elif sp_dist == 2:
                x[0,:] = xmean
                
                for i in range(nem_sclfct):
                    x[1+(2*nem_sclfct*n)+i,:] = xmean + nrml_bmt[i]*sigma_mtx[:,n]
                    x[1+(2*nem_sclfct*n)+nem_sclfct+i,:] = xmean - nrml_bmt[i]*sigma_mtx[:,n]                        

    xptb = x - xmean 
    
    return xmean, xptb, np.diag(P_b), np.diag(P), zrsd, beta_mf

def getkf(xmean,xptb,h,obs,obs_r):
    """GETKF (use only with full rank ensemble, no localization)"""
    nanals, ndim = xptb.shape; nobs = obs.shape[-1]
    # forward operator.
    zptb = np.empty((nanals, nobs), xptb.dtype)
    for nanal in range(nanals):
        zptb[nanal] = np.dot(h,xptb[nanal])
    hxmean = np.dot(h,xmean)
    sqrtoberrvar_inv = 1./np.sqrt(obs_r)
    YbRsqrtinv = zptb * sqrtoberrvar_inv
    u, s, v = svd(YbRsqrtinv,full_matrices=False,lapack_driver='gesvd')
    sp = s**2+nanals-1
    painv =  (u * (1./sp)).dot(u.T)
    kfgain = np.dot(xptb.T,np.dot(painv,YbRsqrtinv*sqrtoberrvar_inv))
    xmean = xmean + np.dot(kfgain, obs-hxmean)
    reducedgain = np.dot(xptb.T,u)*(1.-np.sqrt((nanals-1)/sp))
    # ETKF form
    # method 1
    #pasqrt_inv =  (u * (np.sqrt((nanals-1)/sp))).dot(u.T)
    #xptb = np.dot(xptb.T, pasqrt_inv).T
    # method 2
    #xptb = xptb - np.dot(reducedgain,u.T).T
    # this is equivalent to above, since u.T = np.dot((v.T/s).T,YbRsqrtinv.T)
    #xptb = xptb - np.dot(reducedgain,np.dot((v.T/s).T,YbRsqrtinv.T)).T
    # GETKF form
    reducedgain = np.dot(reducedgain,(v.T/s).T)*sqrtoberrvar_inv
    xptb = xptb - np.dot(reducedgain,zptb.T).T
    return xmean, xptb, 1.

def getkf_modens(xmean, xptb, h, h_type, obs, obs_r, z):
    """GETKF with modulated ensemble"""
    nems, ndim = xptb.shape
    nobs = obs.shape[-1]
    svd_calc = True
    
    if z is None:
        raise ValueError('z not specified')                                         # z = W^T
        
    # modulation ensemble
    neig = z.shape[0]                                                               # number of eigenvalues
    nems_modens = neig*nems
    iens_modens = 0

    xptb_modens = np.zeros((nems_modens,ndim),xptb.dtype)
    
    for j in range(neig):
        for iens in range(nems):
            xptb_modens[iens_modens,:] = xptb[iens,:]*z[neig-j-1,:]
            #xptb_modens[iens_modens,:] = xptb[iens,:]*z[j,:]                       # same as upper line            
            iens_modens += 1
            
    xptb_modens = np.sqrt(float(nems_modens-1)/float(nems-1))*xptb_modens
        
    # data assimilation
    if h_type == 0:                                                                 # linear h(x)=x (for simulation)
        zptb = np.empty((nems, nobs), xptb.dtype)
        zptb_modens = np.empty((nems_modens, nobs), xptb_modens.dtype)

        for iens in range(nems):
            zptb[iens] = np.dot(h,xptb[iens])
            zptb[iens] = nonlinear_h(zptb[iens], h_type)
            
        for iens in range(nems_modens):
            zptb_modens[iens] = np.dot(h,xptb_modens[iens])
            zptb_modens[iens] = nonlinear_h(zptb_modens[iens], h_type)

        zmean_modens = np.dot(h,xmean)
        zmean_modens = nonlinear_h(zmean_modens, h_type)
        zrsd_modens = obs - zmean_modens
    else:                                                                           # nonlinear h(x)=|x| or nonlinear h(x)=ln(|x|) (for operational mode)
        x = xmean + xptb
        z = np.empty((nems, nobs), xptb.dtype)
    
        for iens in range(nems):
            z[iens] = np.dot(h, x[iens])
            z[iens] = nonlinear_h(z[iens], h_type)
       
        zmean = z.mean(axis=0)
        zptb = z - zmean   
        
        x_modens = xmean + xptb_modens
        z_modens = np.empty((nems_modens, nobs), x_modens.dtype)

        for iens in range(nems_modens):
            z_modens[iens] = np.dot(h, x_modens[iens])
            z_modens[iens] = nonlinear_h(z_modens[iens], h_type)

        zmean_modens = z_modens.mean(axis=0)
        zptb_modens = z_modens - zmean_modens
        zrsd_modens = obs - zmean_modens 

    if svd_calc:                                                                    # singular value decomposition (for simulation)
        Rsqrt_inv = 1./np.sqrt(obs_r)                                               # R^(-1/2)
        YbRsqrtinv = zptb_modens * Rsqrt_inv                                        # (R^(-1/2)*H*Xp)^T = (Yp*H)^T*R^(-1/2)
        u, s, v = svd(YbRsqrtinv,full_matrices=False,lapack_driver='gesvd')         # u, v: orthogonal matrix, s: singular values
        sp = (nems_modens-1) + (s**2)
        painv =  (u*(1./sp)).dot(u.T)
        
        kfgain = np.dot(xptb_modens.T, np.dot(painv, YbRsqrtinv*Rsqrt_inv))         # kalman gain
        xmean = xmean + np.dot(kfgain, zrsd_modens)                                 # analysis mean
        
        reducedgain = np.dot(xptb_modens.T, u)*(1.-np.sqrt((nems_modens-1)/sp))     # reduced kalman gain (modified kalman gain)
        reducedgain = np.dot(reducedgain, (v.T/s).T)*Rsqrt_inv
    else:                                                                           # eigenvalue decompostion (for operational mode)
        Rinv = 1./obs_r                                                             # R^(-1)
        YbRinv = zptb_modens*Rinv                                                   # Yp^T*R^(-1)
        a = np.dot(YbRinv, zptb_modens.T)                                           # Yp^T*R^(-1)*Yp
        evals, evecs = np.linalg.eigh(a)                                            # evals: eigenvalue, evecs: eigenvector
        
        b = (nems_modens-1) + evals
        painv =  np.dot(evecs*(1./b), evecs.T)
        
        kfgain = np.dot(xptb_modens.T, np.dot(painv, YbRinv))                       # kalman gain
        xmean = xmean + np.dot(kfgain, zrsd_modens)                                 # analysis mean
        
        reducedgain = np.dot(xptb_modens.T, evecs)*(1.-np.sqrt((nems_modens-1)/b))*(1./evals)   # reduced kalman gain (modified kalman gain)
        reducedgain = np.dot(reducedgain, np.dot(evecs.T, YbRinv))
        
    xptb = xptb - np.dot(reducedgain, zptb.T).T                                     # analysis ensemble perturbation

    return xmean, xptb, 1.

def getkf_modens_rloc(xmean, xptb, h, h_type, obs, obs_r, obs_rloc, z, mpi, comm, myrank):
    """GETKF with modulated ensemble (B localization) and R localization """
    """
    if mpi:
        comm.Bcast(xmean, root=0)    
        comm.Bcast(xptb, root=0)
        comm.Bcast(obs, root=0)
    """
    
    nems, ndim = xptb.shape
    nobs = obs.shape[-1]
    svd_calc = True
    
    if z is None:
        raise ValueError('z not specified')                                         # z = W^T
        
    # modulation ensemble
    neig = z.shape[0]                                                               # number of eigenvalues
    nems_modens = neig*nems
    iens_modens = 0

    xptb_modens = np.zeros((nems_modens,ndim),xptb.dtype)

    for j in range(neig):
        for iens in range(nems):
            xptb_modens[iens_modens,:] = xptb[iens,:]*z[neig-j-1,:]
            #xptb_modens[iens_modens,:] = xptb[iens,:]*z[j,:]                       # same as upper line            
            iens_modens += 1
            
    xptb_modens = np.sqrt(float(nems_modens-1)/float(nems-1))*xptb_modens    

    # data assimilation
    if h_type == 0:                                                                 # linear h(x)=x (for simulation)
        zptb = np.empty((nems, nobs), xptb.dtype)
        zptb_modens = np.empty((nems_modens, nobs), xptb_modens.dtype)

        for iens in range(nems):
            zptb[iens] = np.dot(h,xptb[iens])
            zptb[iens] = nonlinear_h(zptb[iens], h_type)
            
        for iens in range(nems_modens):
            zptb_modens[iens] = np.dot(h,xptb_modens[iens])
            zptb_modens[iens] = nonlinear_h(zptb_modens[iens], h_type)

        zmean_modens = np.dot(h,xmean)
        zmean_modens = nonlinear_h(zmean_modens, h_type)
        zrsd_modens = obs - zmean_modens
    else:                                                                           # nonlinear h(x)=|x| or nonlinear h(x)=ln(|x|) (for operational mode)
        x = xmean + xptb
        z = np.empty((nems, nobs), xptb.dtype)
    
        for iens in range(nems):
            z[iens] = np.dot(h, x[iens])
            z[iens] = nonlinear_h(z[iens], h_type)
       
        zmean = z.mean(axis=0)
        zptb = z - zmean   
        
        x_modens = xmean + xptb_modens
        z_modens = np.empty((nems_modens, nobs), x_modens.dtype)

        for iens in range(nems_modens):
            z_modens[iens] = np.dot(h, x_modens[iens])
            z_modens[iens] = nonlinear_h(z_modens[iens], h_type)

        zmean_modens = z_modens.mean(axis=0)
        zptb_modens = z_modens - zmean_modens
        zrsd_modens = obs - zmean_modens 
    
    obs_rloc = np.where(obs_rloc < 1.e-13, 1.e-13, obs_rloc)                                    # e.g., x = array([[0.,1.,2.],[3.,4.,5.],[6.,7.,8.]]), y = np.where(x < 5, x, -1) => y = array([[ 0.,1.,2.],[ 3.,4.,-1.],[-1.,-1.,-1.]]) 

    if svd_calc:                                                                                # singular value decomposition (for simulation)
        if mpi:
            xptb_T = np.empty((ndim,nems), float)
            n = myrank                                                      
            if n < ndim:
                Rsqrt_inv = np.sqrt(obs_rloc[n,:]/obs_r)                                        # R^(-1/2)
                YbRsqrtinv = zptb_modens * Rsqrt_inv                                            # (R^(-1/2)*H*Xp)^T = (Yp*H)^T*R^(-1/2)
                u, s, v = svd(YbRsqrtinv,full_matrices=False,lapack_driver='gesvd')             # u, v: orthogonal matrix, s: singular values
                sp = (nems_modens-1) + (s**2)
                painv =  (u*(1./sp)).dot(u.T)
                
                kfgain = np.dot(xptb_modens[:,n].T, np.dot(painv, YbRsqrtinv*Rsqrt_inv))        # kalman gain
                xmean[n] = xmean[n] + np.dot(kfgain, zrsd_modens)                               # analysis mean
                
                reducedgain = np.dot(xptb_modens[:,n].T, u)*(1.-np.sqrt((nems_modens-1)/sp))    # reduced kalman gain (modified kalman gain)
                reducedgain = np.dot(reducedgain, (v.T/s).T)*Rsqrt_inv
                xptb[:,n] = xptb[:,n] - np.dot(reducedgain, zptb.T).T                           # analysis ensemble perturbation     

                comm.Allgather(np.array(xmean[n]), xmean)
                comm.Allgather(np.array(xptb[:,n]), xptb_T)
                xptb = xptb_T.T                
        else:
            for n in range(ndim): 
                Rsqrt_inv = np.sqrt(obs_rloc[n,:]/obs_r)                                        # R^(-1/2)
                YbRsqrtinv = zptb_modens * Rsqrt_inv                                            # (R^(-1/2)*H*Xp)^T = (Yp*H)^T*R^(-1/2)
                u, s, v = svd(YbRsqrtinv,full_matrices=False,lapack_driver='gesvd')             # u, v: orthogonal matrix, s: singular values
                sp = (nems_modens-1) + (s**2)
                painv =  (u*(1./sp)).dot(u.T)
                
                kfgain = np.dot(xptb_modens[:,n].T, np.dot(painv, YbRsqrtinv*Rsqrt_inv))        # kalman gain
                xmean[n] = xmean[n] + np.dot(kfgain, zrsd_modens)                               # analysis mean
                
                reducedgain = np.dot(xptb_modens[:,n].T, u)*(1.-np.sqrt((nems_modens-1)/sp))    # reduced kalman gain (modified kalman gain)
                reducedgain = np.dot(reducedgain, (v.T/s).T)*Rsqrt_inv
                xptb[:,n] = xptb[:,n] - np.dot(reducedgain, zptb.T).T                           # analysis ensemble perturbation
    else:                                                                                       # eigenvalue decompostion (for operational mode)
        if mpi:
            xptb_T = np.empty((ndim,nems), float)
            n = myrank                                                      
            if n < ndim:
                Rinv = obs_rloc[n,:]/obs_r                                                      # R^(-1)
                YbRinv = zptb_modens*Rinv                                                       # Yp^T*R^(-1)
                a = np.dot(YbRinv, zptb_modens.T)                                               # Yp^T*R^(-1)*Yp
                evals, evecs = np.linalg.eigh(a)                                                # evals: eigenvalue, evecs: eigenvector
                
                b = (nems_modens-1) + evals
                painv =  np.dot(evecs*(1./b), evecs.T)
                
                kfgain = np.dot(xptb_modens[:,n].T, np.dot(painv, YbRinv))                      # kalman gain
                xmean[n] = xmean[n] + np.dot(kfgain, zrsd_modens)                               # analysis mean
                
                reducedgain = np.dot(xptb_modens[:,n].T, evecs)*(1.-np.sqrt((nems_modens-1)/b))*(1./evals)   # reduced kalman gain (modified kalman gain)
                reducedgain = np.dot(reducedgain, np.dot(evecs.T, YbRinv))
                xptb[:,n] = xptb[:,n] - np.dot(reducedgain, zptb.T).T                           # analysis ensemble perturbation            

                comm.Allgather(np.array(xmean[n]), xmean)
                comm.Allgather(np.array(xptb[:,n]), xptb_T)
                xptb = xptb_T.T                
        else:
            for n in range(ndim): 
                Rinv = obs_rloc[n,:]/obs_r                                                      # R^(-1)
                YbRinv = zptb_modens*Rinv                                                       # Yp^T*R^(-1)
                a = np.dot(YbRinv, zptb_modens.T)                                               # Yp^T*R^(-1)*Yp
                evals, evecs = np.linalg.eigh(a)                                                # evals: eigenvalue, evecs: eigenvector
                
                b = (nems_modens-1) + evals
                painv =  np.dot(evecs*(1./b), evecs.T)
                
                kfgain = np.dot(xptb_modens[:,n].T, np.dot(painv, YbRinv))                      # kalman gain
                xmean[n] = xmean[n] + np.dot(kfgain, zrsd_modens)                               # analysis mean
                
                reducedgain = np.dot(xptb_modens[:,n].T, evecs)*(1.-np.sqrt((nems_modens-1)/b))*(1./evals)   # reduced kalman gain (modified kalman gain)
                reducedgain = np.dot(reducedgain, np.dot(evecs.T, YbRinv))
                xptb[:,n] = xptb[:,n] - np.dot(reducedgain, zptb.T).T                           # analysis ensemble perturbation
  
    return xmean, xptb, 1.

def getkf_modens_sqrt(xmean, xptb, h, h_type, obs, obs_r, z):
    """GETKF with modulated ensemble based on covariance matrix square root"""
    nems, ndim = xptb.shape
    nobs = obs.shape[-1]
    svd_calc = True
    
    if z is None:
        raise ValueError('z not specified')                                         # z = W^T
        
    # modulation ensemble
    neig = z.shape[0]                                                               # number of eigenvalues
    nems_modens = neig*nems
    iens_modens = 0

    xsqrt = np.sqrt(1./float(nems-1))*xptb
    xsqrt_modens = np.zeros((nems_modens,ndim), xsqrt.dtype)
    
    for j in range(neig):
        for iens in range(nems):
            xsqrt_modens[iens_modens,:] = xsqrt[iens,:]*z[neig-j-1,:]
            #xsqrt_modens[iens_modens,:] = xsqrt[iens,:]*z[j,:]                     # same as upper line  
            iens_modens += 1

    xptb_modens = np.sqrt(float(nems_modens-1)) * xsqrt_modens
    
    # data assimilation
    if h_type == 0:                                                                 # linear h(x)=x (for simulation)
        zptb = np.empty((nems, nobs), xptb.dtype)
        zptb_modens = np.empty((nems_modens, nobs), xptb_modens.dtype)

        for iens in range(nems):
            zptb[iens] = np.dot(h,xptb[iens])
            zptb[iens] = nonlinear_h(zptb[iens], h_type)
            
        for iens in range(nems_modens):
            zptb_modens[iens] = np.dot(h,xptb_modens[iens])
            zptb_modens[iens] = nonlinear_h(zptb_modens[iens], h_type)

        zmean_modens = np.dot(h,xmean)
        zmean_modens = nonlinear_h(zmean_modens, h_type)
        zsqrt_modens = np.sqrt(1./float(nems_modens-1)) * zptb_modens
        zrsd_modens = obs - zmean_modens
    else:                                                                           # nonlinear h(x)=|x| or nonlinear h(x)=ln(|x|) (for operational mode)
        x = xmean + xptb
        z = np.empty((nems, nobs), xptb.dtype)
    
        for iens in range(nems):
            z[iens] = np.dot(h, x[iens])
            z[iens] = nonlinear_h(z[iens], h_type)
       
        zmean = z.mean(axis=0)
        zptb = z - zmean   
        
        x_modens = xmean + xptb_modens
        z_modens = np.empty((nems_modens, nobs), x_modens.dtype)

        for iens in range(nems_modens):
            z_modens[iens] = np.dot(h, x_modens[iens])
            z_modens[iens] = nonlinear_h(z_modens[iens], h_type)

        zmean_modens = z_modens.mean(axis=0)
        zptb_modens = z_modens - zmean_modens
        zsqrt_modens = np.sqrt(1./float(nems_modens-1)) * zptb_modens
        zrsd_modens = obs - zmean_modens 

    if svd_calc:                                                                    # singular value decomposition (for simulation)
        Rsqrt_inv = 1./np.sqrt(obs_r)                                               # R^(-1/2)
        YbRsqrtinv = zsqrt_modens * Rsqrt_inv                                       # (R^(-1/2)*H*Xp)^T = (Yp*H)^T*R^(-1/2)
        u, s, v = svd(YbRsqrtinv,full_matrices=False,lapack_driver='gesvd')         # u, v: orthogonal matrix, s: singular values
        sp = 1. + (s**2)
        painv =  (u*(1./sp)).dot(u.T)
        
        kfgain = np.dot(xsqrt_modens.T, np.dot(painv, YbRsqrtinv*Rsqrt_inv))        # kalman gain
        xmean = xmean + np.dot(kfgain, zrsd_modens)                                 # analysis mean
        
        reducedgain = np.dot(xsqrt_modens.T, u)*(1.-np.sqrt(1./sp))                 # reduced kalman gain (modified kalman gain)
        reducedgain = np.dot(reducedgain, (v.T/s).T)*Rsqrt_inv
    else:                                                                           # eigenvalue decompostion (for operational mode)
        Rinv = 1./obs_r                                                             # R^(-1)
        YbRinv = zsqrt_modens*Rinv                                                  # Yp^T*R^(-1)
        a = np.dot(YbRinv, zsqrt_modens.T)                                          # Yp^T*R^(-1)*Yp
        evals, evecs = np.linalg.eigh(a)                                            # evals: eigenvalue, evecs: eigenvector
        
        b = 1. + evals
        painv =  np.dot(evecs*(1./b), evecs.T)
        
        kfgain = np.dot(xsqrt_modens.T, np.dot(painv, YbRinv))                      # kalman gain
        xmean = xmean + np.dot(kfgain, zrsd_modens)                                 # analysis mean
        
        reducedgain = np.dot(xsqrt_modens.T, evecs)*(1.-np.sqrt(1./b))*(1./evals)   # reduced kalman gain (modified kalman gain)
        reducedgain = np.dot(reducedgain, np.dot(evecs.T, YbRinv))
        
    xptb = xptb - np.dot(reducedgain, zptb.T).T                                     # analysis ensemble perturbation

    return xmean, xptb, 1.        

def getkf_modens_sqrt_rloc(xmean, xptb, h, h_type, obs, obs_r, obs_rloc, z, mpi, comm, myrank):
    """GETKF with modulated ensemble (B localization) and R localization based on covariance matrix square root"""
    """
    if mpi:
        comm.Bcast(xmean, root=0)    
        comm.Bcast(xptb, root=0)
        comm.Bcast(obs, root=0)
    """
    
    nems, ndim = xptb.shape
    nobs = obs.shape[-1]
    svd_calc = False
    
    if z is None:
        raise ValueError('z not specified')                                         # z = W^T
        
    # modulation ensemble
    neig = z.shape[0]                                                               # number of eigenvalues
    nems_modens = neig*nems
    iens_modens = 0

    xsqrt = np.sqrt(1./float(nems-1))*xptb
    xsqrt_modens = np.zeros((nems_modens,ndim), xsqrt.dtype)
    
    for j in range(neig):
        for iens in range(nems):
            xsqrt_modens[iens_modens,:] = xsqrt[iens,:]*z[neig-j-1,:]
            #xsqrt_modens[iens_modens,:] = xsqrt[iens,:]*z[j,:]                     # same as upper line  
            iens_modens += 1

    xptb_modens = np.sqrt(float(nems_modens-1)) * xsqrt_modens
    
    # data assimilation
    if h_type == 0:                                                                 # linear h(x)=x (for simulation)
        zptb = np.empty((nems, nobs), xptb.dtype)
        zptb_modens = np.empty((nems_modens, nobs), xptb_modens.dtype)

        for iens in range(nems):
            zptb[iens] = np.dot(h,xptb[iens])
            zptb[iens] = nonlinear_h(zptb[iens], h_type)
            
        for iens in range(nems_modens):
            zptb_modens[iens] = np.dot(h, xptb_modens[iens])
            zptb_modens[iens] = nonlinear_h(zptb_modens[iens], h_type)

        zmean_modens = np.dot(h,xmean)
        zmean_modens = nonlinear_h(zmean_modens, h_type)
        zsqrt_modens = np.sqrt(1./float(nems_modens-1)) * zptb_modens
        zrsd_modens = obs - zmean_modens
    else:                                                                           # nonlinear h(x)=|x| or nonlinear h(x)=ln(|x|) (for operational mode)
        x = xmean + xptb
        z = np.empty((nems, nobs), xptb.dtype)
    
        for iens in range(nems):
            z[iens] = np.dot(h, x[iens])
            z[iens] = nonlinear_h(z[iens], h_type)
       
        zmean = z.mean(axis=0)
        zptb = z - zmean   
        
        x_modens = xmean + xptb_modens
        z_modens = np.empty((nems_modens, nobs), x_modens.dtype)

        for iens in range(nems_modens):
            z_modens[iens] = np.dot(h, x_modens[iens])
            z_modens[iens] = nonlinear_h(z_modens[iens], h_type)

        zmean_modens = z_modens.mean(axis=0)
        zptb_modens = z_modens - zmean_modens
        zsqrt_modens = np.sqrt(1./float(nems_modens-1)) * zptb_modens
        zrsd_modens = obs - zmean_modens 

    obs_rloc = np.where(obs_rloc < 1.e-13, 1.e-13, obs_rloc)
    
    if svd_calc:                                                                                # singular value decomposition (for simulation)
        if mpi:
            xptb_T = np.empty((ndim,nems), float)
            n = myrank                                                      
            if n < ndim:
                Rsqrt_inv = np.sqrt(obs_rloc[n,:]/obs_r)                                        # R^(-1/2)
                YbRsqrtinv = zsqrt_modens * Rsqrt_inv                                           # (R^(-1/2)*H*Xp)^T = (Yp*H)^T*R^(-1/2)
                u, s, v = svd(YbRsqrtinv,full_matrices=False,lapack_driver='gesvd')             # u, v: orthogonal matrix, s: singular values
                sp = 1. + (s**2)
                painv =  (u*(1./sp)).dot(u.T)
                
                kfgain = np.dot(xsqrt_modens[:,n].T, np.dot(painv, YbRsqrtinv*Rsqrt_inv))       # kalman gain
                xmean[n] = xmean[n] + np.dot(kfgain, zrsd_modens)                               # analysis mean
                
                reducedgain = np.dot(xsqrt_modens[:,n].T, u)*(1.-np.sqrt(1./sp))                # reduced kalman gain (modified kalman gain)
                reducedgain = np.dot(reducedgain, (v.T/s).T)*Rsqrt_inv
                xptb[:,n] = xptb[:,n] - np.dot(reducedgain, zptb.T).T                           # analysis ensemble perturbation

                comm.Allgather(np.array(xmean[n]), xmean)
                comm.Allgather(np.array(xptb[:,n]), xptb_T)
                xptb = xptb_T.T 
        else:
            for n in range(ndim):
                Rsqrt_inv = np.sqrt(obs_rloc[n,:]/obs_r)                                        # R^(-1/2)
                YbRsqrtinv = zsqrt_modens * Rsqrt_inv                                           # (R^(-1/2)*H*Xp)^T = (Yp*H)^T*R^(-1/2)
                u, s, v = svd(YbRsqrtinv,full_matrices=False,lapack_driver='gesvd')             # u, v: orthogonal matrix, s: singular values
                sp = 1. + (s**2)
                painv =  (u*(1./sp)).dot(u.T)
                
                kfgain = np.dot(xsqrt_modens[:,n].T, np.dot(painv, YbRsqrtinv*Rsqrt_inv))       # kalman gain
                xmean[n] = xmean[n] + np.dot(kfgain, zrsd_modens)                               # analysis mean
                
                reducedgain = np.dot(xsqrt_modens[:,n].T, u)*(1.-np.sqrt(1./sp))                # reduced kalman gain (modified kalman gain)
                reducedgain = np.dot(reducedgain, (v.T/s).T)*Rsqrt_inv
                xptb[:,n] = xptb[:,n] - np.dot(reducedgain, zptb.T).T                           # analysis ensemble perturbation
    else:                                                                                       # eigenvalue decompostion (for operational mode)
        if mpi:
            xptb_T = np.empty((ndim,nems), float)
            n = myrank                                                      
            if n < ndim:
                Rinv = obs_rloc[n,:]/obs_r                                                      # R^(-1)
                YbRinv = zsqrt_modens*Rinv                                                      # Yp^T*R^(-1)
                a = np.dot(YbRinv, zsqrt_modens.T)                                              # Yp^T*R^(-1)*Yp
                evals, evecs = np.linalg.eigh(a)                                                # evals: eigenvalue, evecs: eigenvector
                
                b = 1. + evals
                painv =  np.dot(evecs*(1./b), evecs.T)
                
                kfgain = np.dot(xsqrt_modens[:,n].T, np.dot(painv, YbRinv))                     # kalman gain
                xmean[n] = xmean[n] + np.dot(kfgain, zrsd_modens)                               # analysis mean
                
                reducedgain = np.dot(xsqrt_modens[:,n].T, evecs)*(1.-np.sqrt(1./b))*(1./evals)  # reduced kalman gain (modified kalman gain)
                reducedgain = np.dot(reducedgain, np.dot(evecs.T, YbRinv))
                xptb[:,n] = xptb[:,n] - np.dot(reducedgain, zptb.T).T                           # analysis ensemble perturbation

                comm.Allgather(np.array(xmean[n]), xmean)
                comm.Allgather(np.array(xptb[:,n]), xptb_T)
                xptb = xptb_T.T             
        else:
            for n in range(ndim):
                Rinv = obs_rloc[n,:]/obs_r                                                      # R^(-1)
                YbRinv = zsqrt_modens*Rinv                                                      # Yp^T*R^(-1)
                a = np.dot(YbRinv, zsqrt_modens.T)                                              # Yp^T*R^(-1)*Yp
                evals, evecs = np.linalg.eigh(a)                                                # evals: eigenvalue, evecs: eigenvector
                
                b = 1. + evals
                painv =  np.dot(evecs*(1./b), evecs.T)
                
                kfgain = np.dot(xsqrt_modens[:,n].T, np.dot(painv, YbRinv))                     # kalman gain
                xmean[n] = xmean[n] + np.dot(kfgain, zrsd_modens)                               # analysis mean
                
                reducedgain = np.dot(xsqrt_modens[:,n].T, evecs)*(1.-np.sqrt(1./b))*(1./evals)  # reduced kalman gain (modified kalman gain)
                reducedgain = np.dot(reducedgain, np.dot(evecs.T, YbRinv))
                xptb[:,n] = xptb[:,n] - np.dot(reducedgain, zptb.T).T                           # analysis ensemble perturbation

    return xmean, xptb, 1.    
    
def etkf_modens(xmean,xptb,h,obs,obs_r,z,rs=None,po=False,ss=False,adjust_obnoise=False,denkf=False):
    """ETKF with modulated ensemble."""
    nanals, ndim = xptb.shape; nobs = obs.shape[-1]
    if z is None:
        raise ValueError('z not specified')
    # modulation ensemble
    neig = z.shape[0]; nanals2 = neig*nanals; nanal2 = 0
    xprime2 = np.zeros((nanals2,ndim),xptb.dtype)
    for j in range(neig):
        for nanal in range(nanals):
            xprime2[nanal2,:] = xptb[nanal,:]*z[neig-j-1,:]
            nanal2 += 1
    normfact = np.sqrt(float(nanals2-1)/float(nanals-1))
    xprime2 = normfact*xprime2
    #var = ((xptb**2).sum(axis=0)/(nanals-1)).mean()
    #var2 = ((xprime2**2).sum(axis=0)/(nanals2-1)).mean()
    # 1st nanals members are original members multiplied by scalefact
    # (which is proportional to 1st eigenvector of cov local matrix)
    scalefact = normfact*z[-1]
    #import matplotlib.pyplot as plt
    #plt.plot(np.arange(80), z[-1])
    #plt.title('1st eigenvector of localization matrix')
    #plt.xlabel('j')
    #plt.ylabel('eigenvector amplitude')
    #plt.ylim(-1.2,0.2)
    #plt.axhline(0,color='k')
    #plt.savefig('eig1.png')
    #print np.abs(z[-1]).max()/np.abs(z[-1]).min()
    #plt.show()
    #raise SystemExit
    # forward operator.
    zptb = np.empty((nanals2, nobs), xprime2.dtype)
    hxprime_orig = np.empty((nanals, nobs), xptb.dtype)
    for nanal in range(nanals2):
        zptb[nanal] = np.dot(h,xprime2[nanal])
    for nanal in range(nanals):
        hxprime_orig[nanal] = np.dot(h,xptb[nanal])
    hxmean = np.dot(h,xmean)

    YbRinv = np.dot(zptb,(1./obs_r)*np.eye(nobs))
    pa = (nanals2-1)*np.eye(nanals2)+np.dot(YbRinv,zptb.T)
    pasqrt_inv, painv = symsqrtinv_psd(pa)
    kfgain = np.dot(xprime2.T,np.dot(painv,YbRinv))
    xmean = xmean + np.dot(kfgain, obs-hxmean)
    if denkf:
        xptb = xptb - np.dot(0.5*kfgain,hxprime_orig.T).T
    elif po: # use perturbed obs to update ensemble perts
        if rs is None:
            raise ValueError('must pass random state if po=True')
        # generate obnoise, make sure it has zero mean
        obnoise = rs.standard_normal(size=(nanals,nobs))
        obnoise = obnoise - obnoise.mean(axis=0)
        if adjust_obnoise:
            # remove part of obnoise that lies in suspace of zptb
            cxy = np.dot(obnoise, hxprime_orig.T)
            cxx = np.dot(hxprime_orig,hxprime_orig.T)
            # pseudo-inverse of a symmetrix matrix (same as above)
            cxxinv = pinvh(cxx)
            # compute multivariate regression matrix, find part of obnoise that
            # is linearly related to zptb, subtract from obnoise.
            obnoise = obnoise - np.dot(np.dot(cxy,cxxinv), hxprime_orig)
            # make sure mean is still zero
            obnoise = obnoise - obnoise.mean(axis=0)
        # rescale so obnoise has expected variance.
        #obnoise=np.sqrt(obs_r/((obnoise**2).sum(axis=0)/(nanals-1)))*obnoise
        obnoise=np.sqrt(obs_r/(((obnoise**2).sum(axis=0)/(nanals-1))).mean())*obnoise
        # check that cross-covariance really is zero
        #cxy = np.dot(obnoise, hxprime_orig.T)
        #print cxy.min(), cxy.max()
        #raise SystemExit
        zptb = obnoise  + hxprime_orig
        xptb = xptb - np.dot(kfgain,zptb.T).T
    elif ss: # use stochastic subsampling to select posterior perturbations.
        pasqrt_inv, painv = symsqrtinv_psd(pa)
        enswts = np.sqrt(nanals2-1)*pasqrt_inv
        xprime_full = np.dot(enswts.T,xprime2)
        # deterministic sub-sampling
        #xptb = xprime_full[np.random.choice(nanal2, nanals, replace=False)]
        # stochastic sub-sampling (nanals random linear combos of nanals
        # posterior perturbations)
        #print ((xprime_full**2).sum(axis=0)/(nanals2-1)).mean()
        ranwts = rs.standard_normal(size=(nanals,nanals2))/np.sqrt(nanals2-1)
        ranwts_mean = ranwts.mean(axis=1)
        # make sure weights sum to zero and stdev=1
        ranwts = ranwts - ranwts_mean[:,np.newaxis]
        ranwts_stdev = np.sqrt((ranwts**2).sum(axis=1))
        ranwts = ranwts/ranwts_stdev[:,np.newaxis]
        xptb = np.dot(ranwts,xprime_full)
        #print ((xptb**2).sum(axis=0)/(nanals-1)).mean()
        #raise SystemExit
        xptb = xptb - xptb.mean(axis=0)
    else:
        # compute reduced gain to update perts
        #D = np.dot(zptb.T, zptb)/(nanals2-1) + obs_r*np.eye(nobs)
        #Dsqrt = symsqrt_psd(D) # symmetric square root of pos-def sym matrix
        #tmp = Dsqrt + np.sqrt(obs_r)*np.eye(nobs)
        #tmpinv = cho_solve(cho_factor(tmp),np.eye(nobs))
        #gainfact = np.dot(Dsqrt,tmpinv)
        #kfgain = np.dot(kfgain, gainfact)
        #zptb = zptb[0:nanals]/scalefact
        #xptb = xptb - np.dot(kfgain,zptb.T).T
        # update modulated ensemble perts with ETKF weights
        pasqrt_inv, painv = symsqrtinv_psd(pa)
        enswts = np.sqrt(nanals2-1)*pasqrt_inv
        # just use 1st nanals posterior members, rescaled.
        #xprime2 = np.dot(enswts.T,xprime2)
        #xptb = xprime2[0:nanals]/scalefact
        # this is equivalent, but a little faster
        xptb = np.dot(enswts[:,0:nanals].T,xprime2)/scalefact
        #xprime_mean = np.abs(xptb.mean(axis=0))
        #xptb = xptb-xprime_mean
        # make sure mean of posterior perts is zero
        #if xprime_mean.max() > 1.e-6:
        #    raise ValueError('nonzero perturbation mean')

    return xmean, xptb, 1.

def letkf(xmean, xptb, h, h_type, obs, obs_r, obs_rloc, mpi, comm, myrank):
    """LETKF (with observation localization)"""
    """
    if mpi:
        comm.Bcast(xmean, root=0)    
        comm.Bcast(xptb, root=0)
        comm.Bcast(obs, root=0)
    """
    
    nems, ndim = xptb.shape                                                     # xptb: nems x ndim
    nobs = obs.shape[-1]                                                        # obs: 1 x nobs(ndim)
    zptb = np.zeros((nems, nobs), xptb.dtype)

    """
    for iens in range(nems):
        zptb[iens] = np.dot(h, xptb[iens])
        zptb[iens] = nonlinear_h(zptb[iens], h_type)

    zmean = np.dot(h, xmean)
    zmean = nonlinear_h(zmean, h_type)
    zrsd = obs - zmean
    """
    
    x = xmean + xptb
    z = np.zeros((nems, nobs), x.dtype)
    
    for iens in range(nems):
        z[iens] = np.dot(h, x[iens])
        z[iens] = nonlinear_h(z[iens], h_type)
   
    zmean = z.mean(axis=0)
    zptb = z - zmean
    zrsd = obs - zmean    
    
    obs_rloc = np.where(obs_rloc < 1.e-13, 1.e-13, obs_rloc)                    # e.g., x = array([[0.,1.,2.],[3.,4.,5.],[6.,7.,8.]]), y = np.where(x < 5, x, -1) => y = array([[ 0.,1.,2.],[ 3.,4.,-1.],[-1.,-1.,-1.]]) 
    
    # calculate analysis mean and perturbation of each grid (each grid has one state variable) for local data assimilation (refer to ETKF for global data assimilation)
    if mpi:
        xptb_T = np.empty((ndim,nems), float)
        n = myrank                                                      
        if n < ndim:
            Rinv = np.diag(obs_rloc[n,:]/obs_r)
            C = np.dot(zptb, Rinv)                                                  # zptb: (Yb)^T
            sqrt_pa, pa = symsqrtinv_psd((nems-1)*np.eye(nems)+np.dot(C,zptb.T))    # symmetric matrix inverse and its square root using eigen decomposition
            K = np.dot(xptb[:,n].T, np.dot(pa,C))          
            Wa = np.sqrt(nems-1)*sqrt_pa
            xmean[n] = xmean[n] + np.dot(K, zrsd)
            xptb[:,n] = np.dot(Wa.T, xptb[:,n])       

            comm.Allgather(np.array(xmean[n]), xmean)
            comm.Allgather(np.array(xptb[:,n]), xptb_T)
            xptb = xptb_T.T
    else:
        for n in range(ndim):                                                       # sequential operation for each state (as in sequential KF)
            Rinv = np.diag(obs_rloc[n,:]/obs_r)
            C = np.dot(zptb, Rinv)                                                  # zptb: (Yb)^T
            sqrt_pa, pa = symsqrtinv_psd((nems-1)*np.eye(nems)+np.dot(C,zptb.T))    # symmetric matrix inverse and its square root using eigen decomposition
            K = np.dot(xptb[:,n].T, np.dot(pa,C))          
            Wa = np.sqrt(nems-1)*sqrt_pa
            xmean[n] = xmean[n] + np.dot(K, zrsd)
            xptb[:,n] = np.dot(Wa.T, xptb[:,n])
   
    beta_mf = np.empty(ndim, float)
    beta_mf[:] = 1.                                                                 # weight for Gaussian sum filter
    
    return xmean, xptb, zrsd, beta_mf
    
def lutkf(x, xmean, xptb, proc_q, proc_q_covinfl_par, h, h_type, obs, obs_r, obs_rloc, ndim_loc, nem_sclfct, sp_dist, nrml_bmt, K_solver, mpi, comm, myrank):
    """LUTKF (with observation localization)"""
    if mpi:
        #comm.Bcast(x, root=0) 
        comm.Bcast(xmean, root=0)    
        comm.Bcast(xptb, root=0)
        comm.Bcast(obs, root=0)
        
    nem, ndim = x.shape                                                             # x: nem x ndim
    nobs = obs.shape[-1]                                                            # obs: 1 x nobs(ndim)

    w = 1./(2.**nem_sclfct)

    P_b = np.zeros(ndim, float)                                                  # background error covariance 
    P = np.zeros(ndim, float)                                                    # error covariance                                      
    xptb_w = np.sqrt(w) * xptb
    Q = proc_q * proc_q_covinfl_par

    if mpi:
        n = myrank                                                      
        if n < ndim:
            P[n] = np.dot(xptb_w[:,n].T, xptb_w[:,n]) + Q
            comm.Allgather(np.array(P[n]), P)
    else:              
        for n in range(ndim):                                                       # sequential operation for each state (as in sequential KF)
            P[n] = np.dot(xptb_w[:,n].T, xptb_w[:,n]) + Q

    P_b = P.copy()
    
    z = np.zeros((nem, nobs), x.dtype)
    zptb = np.zeros((nem, nobs), xptb.dtype)
    
    for iem in range(nem):
        z[iem] = np.dot(h, x[iem])
        z[iem] = nonlinear_h(z[iem], h_type)
            
    zmean = z.mean(axis=0)
    zptb = z - zmean
    zptb_w = np.sqrt(w) * zptb
    zrsd = obs - zmean
    
    obs_rloc = np.where(obs_rloc < 1.e-13, 1.e-13, obs_rloc)                        # e.g., x = array([[0.,1.,2.],[3.,4.,5.],[6.,7.,8.]]), y = np.where(x < 5, x, -1) => y = array([[ 0.,1.,2.],[ 3.,4.,-1.],[-1.,-1.,-1.]]) 
    
    beta_mf = np.empty(ndim, float)
    
    # calculate analysis mean and perturbation of each grid (each grid has one state variable) for local data assimilation (refer to UTKF for global data assimilation)
    if mpi:
        n = myrank                                                      
        if n < ndim:
            if K_solver in [0,1,2]:
                R = np.diag(obs_r/obs_rloc[n,:])
                S = np.dot(zptb_w.T, zptb_w) + R
                Pxz = np.dot(xptb_w[:,n].T, zptb_w)

            if K_solver == 0:                                                       # 0: linear solver using matrix inverse (slow)
                Sinv = np.linalg.inv(S)
                beta_mf[n] = np.exp(-0.5*np.dot(np.dot(zrsd.T, Sinv), zrsd))        # weight for Gaussian sum filter
                K = np.dot(Pxz, Sinv)                                               # K = PxzS^(-1)
                xmean[n] = xmean[n] + np.dot(K, zrsd)
                P[n] = P[n] - np.dot(Pxz, K.T)
            elif K_solver in [1,2]:                                                 # 1: standard linear solver (fast, safe), 2: linear solver using cholesky factorization(fastest, unsafe)
                beta_mf[n] = 1.                                                     # weight for Gaussian sum filter
                K_t = linalg_solve(S, Pxz.T, K_solver)                              # SK^t=Pxz^t
                xmean[n] = xmean[n] + np.dot(K_t.T, zrsd)
                P[n] = P[n] - np.dot(Pxz, K_t)
            elif K_solver == 3:                                                     # 3: computation in the subspace spanned by the ensemble (fastest, safe)
                beta_mf[n] = 1. 
                Rinv = np.diag(obs_rloc[n,:]/obs_r)
                C = np.dot(zptb_w, Rinv)                                            # zptb: (Yb)^T
                pa = syminv_psd(np.eye(nem)+np.dot(C,zptb_w.T))                        # symmetric matrix inverse and its square root using eigen decomposition
                K = np.dot(xptb_w[:,n].T, np.dot(pa,C))
                xmean[n] = xmean[n] + np.dot(K, zrsd)
                P[n] = np.dot(xptb_w[:,n].T, np.dot(pa,xptb_w[:,n])) + Q

            comm.Allgather(np.array(beta_mf[n]), beta_mf)
            comm.Allgather(np.array(xmean[n]), xmean)
            comm.Allgather(np.array(P[n]), P)
    else:
        for n in range(ndim):                                                       # sequential operation for each state (as in sequential KF)
            if K_solver in [0,1,2]:
                R = np.diag(obs_r/obs_rloc[n,:])
                S = np.dot(zptb_w.T, zptb_w) + R
                Pxz = np.dot(xptb_w[:,n].T, zptb_w)
            
            if K_solver == 0:                                                       # 0: linear solver using matrix inverse (slow)
                Sinv = np.linalg.inv(S)
                beta_mf[n] = np.exp(-0.5*np.dot(np.dot(zrsd.T, Sinv), zrsd))        # weight for Gaussian sum filter
                K = np.dot(Pxz, Sinv)                                               # K = PxzS^(-1)
                xmean[n] = xmean[n] + np.dot(K, zrsd)
                P[n] = P[n] - np.dot(Pxz, K.T)
            elif K_solver in [1,2]:                                                 # 1: standard linear solver (fast, safe), 2: linear solver using cholesky factorization(fastest, unsafe)
                beta_mf[n] = 1.                                                     # weight for Gaussian sum filter
                K_t = linalg_solve(S, Pxz.T, K_solver)                              # SK^t=Pxz^t
                xmean[n] = xmean[n] + np.dot(K_t.T, zrsd)
                P[n] = P[n] - np.dot(Pxz, K_t)   
            elif K_solver == 3:                                                     # 3: computation in the subspace spanned by the ensemble (fastest, safe)
                beta_mf[n] = 1. 
                Rinv = np.diag(obs_rloc[n,:]/obs_r)
                C = np.dot(zptb_w, Rinv)                                            # zptb: (Yb)^T
                pa = syminv_psd(np.eye(nem)+np.dot(C,zptb_w.T))                        # symmetric matrix inverse and its square root using eigen decomposition
                K = np.dot(xptb_w[:,n].T, np.dot(pa,C))
                xmean[n] = xmean[n] + np.dot(K, zrsd)
                P[n] = np.dot(xptb_w[:,n].T, np.dot(pa,xptb_w[:,n])) + Q

    beta_mf = np.where(beta_mf < 1.e-13, 1.e-13, beta_mf) 
      
    # sigma point selection
    if mpi:
        x_T = np.empty((ndim, nem), float)
        n = myrank                                                      
        if n < ndim:
            sigma = np.sqrt(float(ndim_loc)*P[n])

            if sp_dist == 0:                                                        # uniform distribution
                for i in range(nem_sclfct):     
                    x[i,n] = xmean[n] + (i+1)/nem_sclfct*sigma
                    x[nem_sclfct+i,n] = xmean[n] - (i+1)/nem_sclfct*sigma
            elif sp_dist == 1:                                                      # normal distribution
                if nem_sclfct > 1:
                    n_rng = get_truncated_normal(0, np.sqrt(P[n]), 0, sigma)
                    rn_list = n_rng.rvs(nem_sclfct - 1)
                    
                for i in range(nem_sclfct):
                    if i == np.arange(nem_sclfct).max():
                        x[i,n] = xmean[n] + sigma
                        x[nem_sclfct+i,n] = xmean[n] - sigma
                    else:
                        x[i,n] = xmean[n] + rn_list[i]
                        x[nem_sclfct+i,n] = xmean[n] - rn_list[i]
            elif sp_dist == 2:                                                      # normal distribution using box muller transform
                for i in range(nem_sclfct):     
                    x[i,n] = xmean[n] + nrml_bmt[i]*sigma
                    x[nem_sclfct+i,n] = xmean[n] - nrml_bmt[i]*sigma                        
                        
            comm.Allgather(np.array(x[:,n]), x_T)
            x = x_T.T
    else:
        for n in range(ndim):
            sigma = np.sqrt(float(ndim_loc)*P[n])

            if sp_dist == 0:                                                        # uniform distribution
                for i in range(nem_sclfct):     
                    x[i,n] = xmean[n] + (i+1)/nem_sclfct*sigma
                    x[nem_sclfct+i,n] = xmean[n] - (i+1)/nem_sclfct*sigma
            elif sp_dist == 1:                                                      # normal distribution
                if nem_sclfct > 1:
                    n_rng = get_truncated_normal(0, np.sqrt(P[n]), 0, sigma)
                    rn_list = n_rng.rvs(nem_sclfct - 1)
                    
                for i in range(nem_sclfct):
                    if i == np.arange(nem_sclfct).max():
                        x[i,n] = xmean[n] + sigma
                        x[nem_sclfct+i,n] = xmean[n] - sigma
                    else:
                        x[i,n] = xmean[n] + rn_list[i]
                        x[nem_sclfct+i,n] = xmean[n] - rn_list[i]
            elif sp_dist == 2:                                                      # normal distribution using box muller transform
                for i in range(nem_sclfct):     
                    x[i,n] = xmean[n] + nrml_bmt[i]*sigma
                    x[nem_sclfct+i,n] = xmean[n] - nrml_bmt[i]*sigma                  
            
    xptb = x - xmean
            
    return xmean, xptb, P_b, P, zrsd, beta_mf
    
def slutkf(x, xmean, xptb, proc_q, proc_q_covinfl_par, h, h_type, obs, obs_r, obs_rloc, ndim_loc, en, alpha, beta, kappa, lamda, nem_sclfct, sp_dist, nrml_bmt, K_solver, mpi, comm, myrank):
    """Scaled-LUTKF (with observation localization)"""
    if mpi:
        #comm.Bcast(x, root=0) 
        comm.Bcast(xmean, root=0)    
        comm.Bcast(xptb, root=0)
        comm.Bcast(obs, root=0)
        
    nem, ndim = x.shape                                                             # x: nem x ndim
    nobs = obs.shape[-1]                                                            # obs: 1 x nobs(ndim)
    
    wm = np.empty(nem, float)
    wc = np.empty(nem, float)

    wm[0] = lamda / (float(en) + lamda)
    wc[0] = (lamda / (float(en) + lamda)) + (1. - (alpha**2) + beta)
    wm[1:nem+1] = wc[1:nem+1] = 1. / (2.*(float(en) + lamda))
    
    xmean = np.sum(x.T * wm, axis=1)     
    xptb = x - xmean
    xptb_w = (np.sqrt(wc) * xptb.T).T
    Q = proc_q * proc_q_covinfl_par
    
    P = np.zeros(ndim, float)                                                    # background error covariance                                      

    if mpi:
        n = myrank                                                      
        if n < ndim:
            P[n] = np.dot(xptb_w[:,n].T, xptb_w[:,n]) + Q
            comm.Allgather(np.array(P[n]), P)
    else:        
        for n in range(ndim):                                                       # sequential operation for each state (as in sequential KF)
            P[n] = np.dot(xptb_w[:,n].T, xptb_w[:,n]) + Q

    P_b = P.copy()
    
    z = np.zeros((nem, nobs), x.dtype)
    zptb = np.zeros((nem, nobs), xptb.dtype)
    
    for iem in range(nem):
        z[iem] = np.dot(h, x[iem])
        z[iem] = nonlinear_h(z[iem], h_type)
            
    zmean = np.sum(z.T * wm, axis=1)    
    zptb = z - zmean
    zptb_w = (np.sqrt(wc) * zptb.T).T
    zrsd = obs - zmean
    
    obs_rloc = np.where(obs_rloc < 1.e-13, 1.e-13, obs_rloc)                        # e.g., x = array([[0.,1.,2.],[3.,4.,5.],[6.,7.,8.]]), y = np.where(x < 5, x, -1) => y = array([[ 0.,1.,2.],[ 3.,4.,-1.],[-1.,-1.,-1.]]) 

    beta_mf = np.empty(ndim, float)
    
    # calculate analysis mean and perturbation of each grid (each grid has one state variable) for local data assimilation (refer to SUTKF for global data assimilation)
    if mpi:
        n = myrank                                                      
        if n < ndim:
            if K_solver in [0,1,2]:
                R = np.diag(obs_r/obs_rloc[n,:])
                S = np.dot(zptb_w.T, zptb_w) + R
                Pxz = np.dot(xptb_w[:,n].T, zptb_w)

            if K_solver == 0:                                                       # 0: linear solver using matrix inverse (slow)
                Sinv = np.linalg.inv(S)
                beta_mf[n] = np.exp(-0.5*np.dot(np.dot(zrsd.T, Sinv), zrsd))        # weight for Gaussian sum filter        
                K = np.dot(Pxz, Sinv)                                               # K = PxzS^(-1)
                xmean[n] = xmean[n] + np.dot(K, zrsd)
                P[n] = P[n] - np.dot(Pxz, K.T)
            elif K_solver in [1,2]:                                                 # 1: standard linear solver (fast, safe), 2: linear solver using cholesky factorization(fastest, unsafe)
                beta_mf[n] = 1.                                                     # weight for Gaussian sum filter
                K_t = linalg_solve(S, Pxz.T, K_solver)                              # SK^t=Pxz^t
                xmean[n] = xmean[n] + np.dot(K_t.T, zrsd)
                P[n] = P[n] - np.dot(Pxz, K_t)
            elif K_solver == 3:                                                     # 3: computation in the subspace spanned by the ensemble (fastest, safe)
                beta_mf[n] = 1. 
                Rinv = np.diag(obs_rloc[n,:]/obs_r)
                C = np.dot(zptb_w, Rinv)                                            # zptb: (Yb)^T
                pa = syminv_psd(np.eye(nem)+np.dot(C,zptb_w.T))                        # symmetric matrix inverse using eigen decomposition
                K = np.dot(xptb_w[:,n].T, np.dot(pa,C))
                xmean[n] = xmean[n] + np.dot(K, zrsd)
                P[n] = np.dot(xptb_w[:,n].T, np.dot(pa,xptb_w[:,n])) + Q
            
            comm.Allgather(np.array(beta_mf[n]), beta_mf)
            comm.Allgather(np.array(xmean[n]), xmean)
            comm.Allgather(np.array(P[n]), P)
    else:
        for n in range(ndim):                                                       # sequential operation for each state (as in sequential KF)                                                   
            if K_solver in [0,1,2]:
                R = np.diag(obs_r/obs_rloc[n,:])
                S = np.dot(zptb_w.T, zptb_w) + R
                Pxz = np.dot(xptb_w[:,n].T, zptb_w)
                     
            if K_solver == 0:                                                       # 0: linear solver using matrix inverse (slow)
                Sinv = np.linalg.inv(S)
                beta_mf[n] = np.exp(-0.5*np.dot(np.dot(zrsd.T, Sinv), zrsd))        # weight for Gaussian sum filter        
                K = np.dot(Pxz, Sinv)                                               # K = PxzS^(-1)
                xmean[n] = xmean[n] + np.dot(K, zrsd)
                P[n] = P[n] - np.dot(Pxz, K.T)
            elif K_solver in [1,2]:                                                 # 1: standard linear solver (fast, safe), 2: linear solver using cholesky factorization(fastest, unsafe)
                beta_mf[n] = 1.                                                     # weight for Gaussian sum filter
                K_t = linalg_solve(S, Pxz.T, K_solver)                              # SK^t=Pxz^t
                xmean[n] = xmean[n] + np.dot(K_t.T, zrsd)
                P[n] = P[n] - np.dot(Pxz, K_t)
            elif K_solver == 3:                                                     # 3: computation in the subspace spanned by the ensemble (fastest, safe)
                beta_mf[n] = 1. 
                Rinv = np.diag(obs_rloc[n,:]/obs_r)
                C = np.dot(zptb_w, Rinv)                                            # zptb: (Yb)^T
                pa = syminv_psd(np.eye(nem)+np.dot(C,zptb_w.T))                        # symmetric matrix inverse using eigen decomposition
                K = np.dot(xptb_w[:,n].T, np.dot(pa,C))
                xmean[n] = xmean[n] + np.dot(K, zrsd)
                P[n] = np.dot(xptb_w[:,n].T, np.dot(pa,xptb_w[:,n])) + Q

    beta_mf = np.where(beta_mf < 1.e-13, 1.e-13, beta_mf)
    
    # sigma point selection    
    if mpi:
        x_T = np.empty((ndim, nem), float)
        n = myrank                                                      
        if n < ndim:
            sigma = np.sqrt((float(en) + lamda)*P[n])

            if sp_dist == 0:                                                        # uniform distribution of ensemble
                x[0,n] = xmean[n]
                
                for i in range(nem_sclfct):
                    x[1+i,n] = xmean[n] + (i+1)/nem_sclfct*sigma
                    x[1+nem_sclfct+i,n] = xmean[n] - (i+1)/nem_sclfct*sigma
            elif sp_dist == 1:                                                      # normal distribution of ensemble
                if nem_sclfct > 1:
                    n_rng = get_truncated_normal(0, np.sqrt(P[n]), 0, sigma)
                    rn_list = n_rng.rvs(nem_sclfct - 1)

                x[0,n] = xmean[n]
                
                for i in range(nem_sclfct):
                    if i == np.arange(nem_sclfct).max():
                        x[1+i,n] = xmean[n] + sigma
                        x[1+nem_sclfct+i,n] = xmean[n] - sigma
                    else:
                        x[1+i,n] = xmean[n] + rn_list[i]
                        x[1+nem_sclfct+i,n] = xmean[n] - rn_list[i]
            elif sp_dist == 2:                                                      # normal distribution of ensemble using box muller transform (nrml_bmt)
                x[0,n] = xmean[n]
                
                for i in range(nem_sclfct):
                    x[1+i,n] = xmean[n] + nrml_bmt[i]*sigma
                    x[1+nem_sclfct+i,n] = xmean[n] - nrml_bmt[i]*sigma                        

            comm.Allgather(np.array(x[:,n]), x_T)
            x = x_T.T
    else:
        for n in range(ndim):
            sigma = np.sqrt((float(en) + lamda)*P[n])

            if sp_dist == 0:
                x[0,n] = xmean[n]
                
                for i in range(nem_sclfct):
                    x[1+i,n] = xmean[n] + (i+1)/nem_sclfct*sigma
                    x[1+nem_sclfct+i,n] = xmean[n] - (i+1)/nem_sclfct*sigma
            elif sp_dist == 1:
                if nem_sclfct > 1:
                    n_rng = get_truncated_normal(0, np.sqrt(P[n]), 0, sigma)
                    rn_list = n_rng.rvs(nem_sclfct - 1)

                x[0,n] = xmean[n]
                
                for i in range(nem_sclfct):
                    if i == np.arange(nem_sclfct).max():
                        x[1+i,n] = xmean[n] + sigma
                        x[1+nem_sclfct+i,n] = xmean[n] - sigma
                    else:
                        x[1+i,n] = xmean[n] + rn_list[i]
                        x[1+nem_sclfct+i,n] = xmean[n] - rn_list[i]    
            elif sp_dist == 2:                                                      # normal distribution of ensemble using box muller transform (nrml_bmt)
                x[0,n] = xmean[n]
                
                for i in range(nem_sclfct):
                    x[1+i,n] = xmean[n] + nrml_bmt[i]*sigma
                    x[1+nem_sclfct+i,n] = xmean[n] - nrml_bmt[i]*sigma     
                    
    xptb = x - xmean 
            
    return xmean, xptb, P_b, P, zrsd, beta_mf

def resample(x, w, rs_mode, rs):
    nem = x.shape[0] 
    
    rs_cnt = np.zeros(nem, int)
    rs_idx = np.zeros(nem, int)
    
    # obtain a count of the number of children each particle has.
    if rs_mode == 0:                                                            # systematic
        j = 0
        k = 0
        w_sum = np.exp(w).sum()
        w_cdf = 0.0
        base = rs.uniform(0, 1.0/float(nem))
        
        while j < nem:
            w_cdf = w_cdf + (np.exp(w[k]) / w_sum)

            while (((base + (float(j) / float(nem))) < w_cdf) and (j < nem)):
                rs_cnt[k] = rs_cnt[k] + 1
                j = j + 1
                
            k = k + 1            
    elif rs_mode == 1:                                                          # stratified
        j = 0
        k = 0
        w_sum = np.exp(w).sum()
        w_cdf = 0.0
        base = rs.uniform(0, 1.0/float(nem))
        
        while j < nem:
            w_cdf = w_cdf + (np.exp(w[k]) / w_sum)

            while (((base + (float(j) / float(nem))) < w_cdf) and (j < nem)):
                rs_cnt[k] = rs_cnt[k] + 1
                j = j + 1
                base = rs.uniform(0, 1.0/float(nem))
                
            k = k + 1      
        
    # collect low-weight partilces into high-weight particles 
    j = 0

    for i in range(nem):
        if rs_cnt[i] > 0:
            rs_idx[i] = i
            while rs_cnt[i] > 1:
                while rs_cnt[j] > 0:
                    j = j + 1
                rs_idx[j] = i
                j = j + 1
                rs_cnt[i] = rs_cnt[i] - 1 
        
    # perform the replication of the chosen
    for i in range(nem):
        if rs_idx[i] != i:
            x[i] = x[rs_idx[i]] 

    # initialize weight of samples    
    w = 0.

def pf(x, xmean, xptb, h, h_type, obs, obs_r, ds_func, nu, rs_thrd, rs_mode, w, w_acc, rs):
    """PF (use only with full rank ensemble, no localization)"""
    nem, ndim = x.shape                                                         # backgournd (x: nem x ndim)
    nobs = obs.shape[-1]                                                        # observation (obs: 1 x nobs(ndim))
    
    z = np.zeros((nem, nobs), x.dtype)
    zrsd = np.zeros((nem, nobs), xptb.dtype)
    
    for iem in range(nem):
        z[iem] = np.dot(h, x[iem])                                              # predicted observation (measurement)
        z[iem] = nonlinear_h(z[iem], h_type)
            
    zrsd = obs - z                                                              # residual (assuming that each predicted observation is predicted observation mean)
    
    # calculate analysis mean and perturbation of each grid (each grid has one state variable) for global data assimilation
    Rinv = (1./obs_r)*np.eye(nobs)
    
    # sample weighting
    wgt = 0.
    
    for iem in range(nem):
        if ds_func == 0:                                                                            # pdf of multivariate t-distribution                                                                       
            #wgt = (1.0 + (zrsd[iem].T @ Rinv @ zrsd[iem])/nu)**(-0.5*(nu + float(nobs)))
            wgt = (-0.5*(nu + float(nobs)))*np.log(1.0 + np.dot(np.dot(zrsd[iem].T, Rinv), zrsd[iem])/nu)       # A @ B is equal to np.dot(A,B)
        elif ds_func == 1:                                                                          # pdf of multivariate normal distribution
            #wgt = np.exp(-0.5*(zrsd[iem].T @ Rinv @ zrsd[iem]))                
            wgt = -0.5*np.dot(np.dot(zrsd[iem].T, Rinv), zrsd[iem])

        if w_acc:
            w[iem] = w[iem] + wgt
        else:
            w[iem] = wgt
            
    w = w - w.max()                                                             # normalize weight of samples (in order that exp(w) has values between 0 and 1)
    w_exp = np.exp(w)
    xmean = (x.T * w_exp/w_exp.sum()).sum(axis=1)

    # resampling
    #ess = (w[:,n].sum()**2)/(w[:,n]**2).sum()
    ess = np.exp(2.0*np.log(w_exp.sum()) - np.log(np.exp(2.0*w).sum()))         # effective sample size
    thrd = rs_thrd * float(nem)                                                 # resampling threshold

    if ess <= thrd:
        resample(x, w, rs_mode, rs)
        
    xptb = x - xmean
    beta_mf = 1.                                                                # weight for Gaussian sum filter
    
    return xmean, xptb, beta_mf

def epf(x, xmean, xptb, h, h_type, obs, obs_r, ds_func, nu, rs_thrd, rs_mode, w, w_acc, rs, mpi, comm, myrank):
    """EPF (extended PF; use only with full rank ensemble, no localization)"""
    if mpi:
        #comm.Bcast(x, root=0) 
        comm.Bcast(xmean, root=0)    
        comm.Bcast(xptb, root=0)
        comm.Bcast(obs, root=0)
        #comm.Bcast(rs, root=0)
        
    nem, ndim = x.shape                                                         # backgournd (x: nem x ndim)
    nobs = obs.shape[-1]                                                        # observation (obs: 1 x nobs(ndim))
    
    z = np.zeros((nem, nobs), x.dtype)
    zrsd = np.zeros((nem, nobs), xptb.dtype)
    
    for iem in range(nem):
        z[iem] = np.dot(h, x[iem])                                              # predicted observation (measurement)
        z[iem] = nonlinear_h(z[iem], h_type)
            
    zrsd = obs - z                                                              # residual (assuming that each predicted observation is predicted observation mean)
    
    # calculate analysis mean and perturbation of each grid (each grid has one state variable) for global data assimilation
    if mpi:
        x_T = np.empty((ndim, nem), float)
        n = myrank                                                      
        if n < ndim:
            Rinv = np.zeros((nobs,nobs), float)
            Rinv[n,n] = (1./obs_r)
            
            # sample weighting
            wgt = 0.
            
            for iem in range(nem):
                if ds_func == 0:                                                    # pdf of multivariate t-distribution                                                                       
                    #wgt = (1.0 + (zrsd[iem].T @ Rinv @ zrsd[iem])/nu)**(-0.5*(nu + float(nobs)))
                    wgt = (-0.5*(nu + float(nobs)))*np.log(1.0 + np.dot(np.dot(zrsd[iem].T, Rinv), zrsd[iem])/nu)       # A @ B is equal to np.dot(A,B)
                elif ds_func == 1:                                                  # pdf of multivariate normal distribution
                    #wgt = np.exp(-0.5*(zrsd[iem].T @ Rinv @ zrsd[iem]))                
                    wgt = -0.5*np.dot(np.dot(zrsd[iem].T, Rinv), zrsd[iem])
                    #wgt = -0.5*((zrsd[iem]**2)* rinv).sum()

                if w_acc:
                    w[iem,n] = w[iem,n] + wgt
                else:
                    w[iem,n] = wgt
                    
            w[:,n] = w[:,n] - w[:,n].max()                                          # normalize weight of samples (in order that exp(w) has values between 0 and 1)
            w_exp = np.exp(w[:,n])
            xmean[n] = (x[:,n] * w_exp/w_exp.sum()).sum()

            # resampling
            #ess = (w[:,n].sum()**2)/(w[:,n]**2).sum()
            ess = np.exp(2.0*np.log(w_exp.sum()) - np.log(np.exp(2.0*w[:,n]).sum()))# effective sample size
            thrd = rs_thrd * float(nem)                                             # resampling threshold

            if ess <= thrd:
                resample(x[:,n], w[:,n], rs_mode, rs)

            comm.Allgather(np.array(xmean[n]), xmean)
            comm.Allgather(np.array(x[:,n]), x_T)
            x = x_T.T
    else:    
        for n in range(ndim):
            Rinv = np.zeros((nobs,nobs), float)
            Rinv[n,n] = (1./obs_r)
            
            # sample weighting
            wgt = 0.
            
            for iem in range(nem):
                if ds_func == 0:                                                    # pdf of multivariate t-distribution                                                                       
                    #wgt = (1.0 + (zrsd[iem].T @ Rinv @ zrsd[iem])/nu)**(-0.5*(nu + float(nobs)))
                    wgt = (-0.5*(nu + float(nobs)))*np.log(1.0 + np.dot(np.dot(zrsd[iem].T, Rinv), zrsd[iem])/nu)       # A @ B is equal to np.dot(A,B)
                elif ds_func == 1:                                                  # pdf of multivariate normal distribution
                    #wgt = np.exp(-0.5*(zrsd[iem].T @ Rinv @ zrsd[iem]))                
                    wgt = -0.5*np.dot(np.dot(zrsd[iem].T, Rinv), zrsd[iem])
                    #wgt = -0.5*((zrsd[iem]**2)* rinv).sum()

                if w_acc:
                    w[iem,n] = w[iem,n] + wgt
                else:
                    w[iem,n] = wgt
                    
            w[:,n] = w[:,n] - w[:,n].max()                                          # normalize weight of samples (in order that exp(w) has values between 0 and 1)
            w_exp = np.exp(w[:,n])
            xmean[n] = (x[:,n] * w_exp/w_exp.sum()).sum()

            # resampling
            #ess = (w[:,n].sum()**2)/(w[:,n]**2).sum()
            ess = np.exp(2.0*np.log(w_exp.sum()) - np.log(np.exp(2.0*w[:,n]).sum()))# effective sample size
            thrd = rs_thrd * float(nem)                                             # resampling threshold

            if ess <= thrd:
                resample(x[:,n], w[:,n], rs_mode, rs)
        
    xptb = x - xmean
    beta_mf = 1.                                                                # weight for Gaussian sum filter
    
    return xmean, xptb, beta_mf
    
def lpf(x, xmean, xptb, h, h_type, obs, obs_r, obs_rloc, ds_func, nu, rs_thrd, rs_mode, w, w_acc, rs, mpi, comm, myrank):
    """LPF (with observation localization)"""
    if mpi:
        #comm.Bcast(x, root=0) 
        comm.Bcast(xmean, root=0)    
        comm.Bcast(xptb, root=0)
        comm.Bcast(obs, root=0)
        #comm.Bcast(rs, root=0)
        
    nem, ndim = x.shape                                                         # backgournd (x: nem x ndim)
    nobs = obs.shape[-1]                                                        # observation (obs: 1 x nobs(ndim))
    
    z = np.zeros((nem, nobs), x.dtype)
    zrsd = np.zeros((nem, nobs), xptb.dtype)
    
    for iem in range(nem):
        z[iem] = np.dot(h, x[iem])                                              # predicted observation (measurement)
        z[iem] = nonlinear_h(z[iem], h_type)
            
    zrsd = obs - z                                                              # residual (assuming that each predicted observation is predicted observation mean)
    
    obs_rloc = np.where(obs_rloc < 1.e-13, 1.e-13, obs_rloc)                    # e.g., x = array([[0.,1.,2.],[3.,4.,5.],[6.,7.,8.]]), y = np.where(x < 5, x, -1) => y = array([[ 0.,1.,2.],[ 3.,4.,-1.],[-1.,-1.,-1.]]) 

    # calculate analysis mean and perturbation of each grid (each grid has one state variable) for local data assimilation (refer to PF for global data assimilation)
    if mpi:
        x_T = np.empty((ndim, nem), float)
        n = myrank                                                      
        if n < ndim:
            Rinv = np.diag(obs_rloc[n,:]/obs_r)
            #rinv = obs_rloc[n,:]/obs_r
            
            # sample weighting
            wgt = 0.
            
            for iem in range(nem):
                if ds_func == 0:                                                    # pdf of multivariate t-distribution                                                                       
                    #wgt = (1.0 + (zrsd[iem].T @ Rinv @ zrsd[iem])/nu)**(-0.5*(nu + float(nobs)))
                    wgt = (-0.5*(nu + float(nobs)))*np.log(1.0 + np.dot(np.dot(zrsd[iem].T, Rinv), zrsd[iem])/nu)       # A @ B is equal to np.dot(A,B)
                elif ds_func == 1:                                                  # pdf of multivariate normal distribution
                    #wgt = np.exp(-0.5*(zrsd[iem].T @ Rinv @ zrsd[iem]))                
                    wgt = -0.5*np.dot(np.dot(zrsd[iem].T, Rinv), zrsd[iem])
                    #wgt = -0.5*((zrsd[iem]**2)* rinv).sum()

                if w_acc:
                    w[iem,n] = w[iem,n] + wgt
                else:
                    w[iem,n] = wgt
                    
            w[:,n] = w[:,n] - w[:,n].max()                                              # normalize weight of samples (in order that exp(w) has values between 0 and 1)
            w_exp = np.exp(w[:,n])
            xmean[n] = (x[:,n] * w_exp/w_exp.sum()).sum()

            # resampling
            #ess = (w[:,n].sum()**2)/(w[:,n]**2).sum()
            ess = np.exp(2.0*np.log(w_exp.sum()) - np.log(np.exp(2.0*w[:,n]).sum()))    # effective sample size
            thrd = rs_thrd * float(nem)                                                 # resampling threshold

            if ess <= thrd:
                resample(x[:,n], w[:,n], rs_mode, rs)
                
            comm.Allgather(np.array(xmean[n]), xmean)
            comm.Allgather(np.array(x[:,n]), x_T)
            x = x_T.T    
    else:                
        for n in range(ndim):
            Rinv = np.diag(obs_rloc[n,:]/obs_r)
            #rinv = obs_rloc[n,:]/obs_r
            
            # sample weighting
            wgt = 0.
            
            for iem in range(nem):
                if ds_func == 0:                                                    # pdf of multivariate t-distribution                                                                       
                    #wgt = (1.0 + (zrsd[iem].T @ Rinv @ zrsd[iem])/nu)**(-0.5*(nu + float(nobs)))
                    wgt = (-0.5*(nu + float(nobs)))*np.log(1.0 + np.dot(np.dot(zrsd[iem].T, Rinv), zrsd[iem])/nu)       # A @ B is equal to np.dot(A,B)
                elif ds_func == 1:                                                  # pdf of multivariate normal distribution
                    #wgt = np.exp(-0.5*(zrsd[iem].T @ Rinv @ zrsd[iem]))                
                    wgt = -0.5*np.dot(np.dot(zrsd[iem].T, Rinv), zrsd[iem])
                    #wgt = -0.5*((zrsd[iem]**2)* rinv).sum()

                if w_acc:
                    w[iem,n] = w[iem,n] + wgt
                else:
                    w[iem,n] = wgt
                    
            w[:,n] = w[:,n] - w[:,n].max()                                              # normalize weight of samples (in order that exp(w) has values between 0 and 1)
            w_exp = np.exp(w[:,n])
            xmean[n] = (x[:,n] * w_exp/w_exp.sum()).sum()

            # resampling
            #ess = (w[:,n].sum()**2)/(w[:,n]**2).sum()
            ess = np.exp(2.0*np.log(w_exp.sum()) - np.log(np.exp(2.0*w[:,n]).sum()))    # effective sample size
            thrd = rs_thrd * float(nem)                                                 # resampling threshold

            if ess <= thrd:
                resample(x[:,n], w[:,n], rs_mode, rs)
        
    xptb = x - xmean

    beta_mf = np.empty(ndim, float)
    beta_mf[:] = 1.                                                                 # weight for Gaussian sum filter
    
    return xmean, xptb, beta_mf


def mlutkf(xmean, xptb, h, h_type, obs, obs_r, obs_rloc, z, nem_sclfct, mpi=False, comm=None, myrank=0):

    nems, ndim = xptb.shape

    if nems > 1:
        var_b = np.var(xptb, axis=0, ddof=1)
        s_b = np.sqrt(np.mean(var_b))
    else:
        s_b = 0.0

    xmean_loc = xmean.copy()
    xptb_loc  = xptb.copy()

    nems, ndim = xptb_loc.shape
    nobs = obs.shape[-1]
    svd_calc = True

    if z is None:
        raise ValueError('z not specified')                                         # z = W^T

    # modulation ensemble
    neig = z.shape[0]                                                               # number of eigenvalues
    nems_modens = neig*nems
    iens_modens = 0

    xptb_modens = np.zeros((nems_modens,ndim),xptb_loc.dtype)

    for j in range(neig):
        for iens in range(nems):
            xptb_modens[iens_modens,:] = xptb_loc[iens,:]*z[neig-j-1,:]
            #xptb_modens[iens_modens,:] = xptb_loc[iens,:]*z[j,:]
            iens_modens += 1

    # xptb_modens = np.sqrt(float(nems_modens-1)/float(nems-1))*xptb_modens
    xptb_modens = np.sqrt(float(nems_modens-1)/float(nems-1))*xptb_modens

    # data assimilation
    if h_type == 0:                                                                 # linear h(x)=x (for simulation)
        zptb = np.empty((nems, nobs), xptb_loc.dtype)
        zptb_modens = np.empty((nems_modens, nobs), xptb_modens.dtype)

        for iens in range(nems):
            zptb[iens] = np.dot(h,xptb_loc[iens])
            zptb[iens] = nonlinear_h(zptb[iens], h_type)

        for iens in range(nems_modens):
            zptb_modens[iens] = np.dot(h,xptb_modens[iens])
            zptb_modens[iens] = nonlinear_h(zptb_modens[iens], h_type)

        zmean_modens = np.dot(h,xmean_loc)
        zmean_modens = nonlinear_h(zmean_modens, h_type)
        zrsd_modens = obs - zmean_modens
    else:                                                                           # nonlinear h(x)
        x = xmean_loc + xptb_loc
        zloc = np.empty((nems, nobs), xptb_loc.dtype)

        for iens in range(nems):
            zloc[iens] = np.dot(h, x[iens])
            zloc[iens] = nonlinear_h(zloc[iens], h_type)

        zmean = zloc.mean(axis=0)
        zptb = zloc - zmean

        x_modens = xmean_loc + xptb_modens
        z_modens = np.empty((nems_modens, nobs), x_modens.dtype)

        for iens in range(nems_modens):
            z_modens[iens] = np.dot(h, x_modens[iens])
            z_modens[iens] = nonlinear_h(z_modens[iens], h_type)

        zmean_modens = z_modens.mean(axis=0)
        zptb_modens = z_modens - zmean_modens
        zrsd_modens = obs - zmean_modens

    obs_rloc = np.where(obs_rloc < 1.e-13, 1.e-13, obs_rloc)
    gamma_max = 1.5

    if svd_calc:                                                                                # SVD
        if mpi:
            n = myrank

            xmean_send = np.array(0.0, dtype=float)               
            xptb_send  = np.zeros((nems,), dtype=float)

            if n < ndim:
                Rsqrt_inv = np.sqrt(obs_rloc[n,:]/obs_r)                                        # R^(-1/2)
                YbRsqrtinv = zptb_modens * Rsqrt_inv
                u, s, v = svd(YbRsqrtinv, full_matrices=False, lapack_driver='gesvd')

                sp = (nems_modens - 1) + (s**2)
                painv = (u * (1.0 / sp)).dot(u.T)

                kfgain = np.dot(xptb_modens[:, n].T, np.dot(painv, YbRsqrtinv * Rsqrt_inv))
                xmean_loc[n] = xmean_loc[n] + np.dot(kfgain, zrsd_modens)

                reducedgain = np.dot(xptb_modens[:, n].T, u) * (1.0 - np.sqrt((nems_modens - 1) / sp))
                reducedgain = np.dot(reducedgain, (v.T / s).T) * Rsqrt_inv
                xptb_loc[:, n] = xptb_loc[:, n] - np.dot(reducedgain, zptb.T).T

                xmean_send = np.array(xmean_loc[n], dtype=float)
                xptb_send[:] = xptb_loc[:, n]

            xmean_all = np.empty((ndim,), dtype=float)
            comm.Allgather(xmean_send, xmean_all)

            xptb_all_cols = np.empty((ndim, nems), dtype=float)
            comm.Allgather(xptb_send, xptb_all_cols)

            xmean_loc[:] = xmean_all
            xptb_loc[:, :] = xptb_all_cols.T
        else:
            for n in range(ndim):
                Rsqrt_inv = np.sqrt(obs_rloc[n,:]/obs_r)
                YbRsqrtinv = zptb_modens * Rsqrt_inv
                u, s, v = svd(YbRsqrtinv,full_matrices=False,lapack_driver='gesvd')
                sp = (nems_modens-1) + (s**2)
                painv =  (u*(1./sp)).dot(u.T)

                kfgain = np.dot(xptb_modens[:,n].T, np.dot(painv, YbRsqrtinv*Rsqrt_inv))
                xmean_loc[n] = xmean_loc[n] + np.dot(kfgain, zrsd_modens)

                reducedgain = np.dot(xptb_modens[:,n].T, u)*(1.-np.sqrt((nems_modens-1)/sp))
                reducedgain = np.dot(reducedgain, (v.T/s).T)*Rsqrt_inv
                xptb_loc[:,n] = xptb_loc[:,n] - np.dot(reducedgain, zptb.T).T
    else:                                                                                       # eigen
        if mpi:
            n = myrank

            xmean_send = np.array(0.0, dtype=float)
            xptb_send  = np.zeros((nems,), dtype=float)

            if n < ndim:
                Rinv   = obs_rloc[n, :] / obs_r
                YbRinv = zptb_modens * Rinv
                a      = np.dot(YbRinv, zptb_modens.T)
                evals, evecs = np.linalg.eigh(a)

                b = (nems_modens - 1) + evals
                painv = np.dot(evecs * (1.0 / b), evecs.T)

                kfgain = np.dot(xptb_modens[:, n].T, np.dot(painv, YbRinv))
                xmean_loc[n] = xmean_loc[n] + np.dot(kfgain, zrsd_modens)

                evals_safe = np.where(evals < 1e-12, 1e-12, evals)

                reducedgain = (np.dot(xptb_modens[:, n].T, evecs)
                               * (1.0 - np.sqrt((nems_modens - 1) / b))
                               * (1.0 / evals_safe))
                reducedgain = np.dot(reducedgain, np.dot(evecs.T, YbRinv))
                xptb_loc[:, n] = xptb_loc[:, n] - np.dot(reducedgain, zptb.T).T

                xmean_send = np.array(xmean_loc[n], dtype=float)
                xptb_send[:] = xptb_loc[:, n]

            xmean_all = np.empty((ndim,), dtype=float)
            comm.Allgather(xmean_send, xmean_all)

            xptb_all_cols = np.empty((ndim, nems), dtype=float)
            comm.Allgather(xptb_send, xptb_all_cols)

            xmean_loc[:] = xmean_all
            xptb_loc[:, :] = xptb_all_cols.T
        else:
            for n in range(ndim):
                Rinv = obs_rloc[n,:]/obs_r
                YbRinv = zptb_modens*Rinv
                a = np.dot(YbRinv, zptb_modens.T)
                evals, evecs = np.linalg.eigh(a)

                b = (nems_modens-1) + evals
                painv =  np.dot(evecs*(1./b), evecs.T)

                kfgain = np.dot(xptb_modens[:,n].T, np.dot(painv, YbRinv))
                xmean_loc[n] = xmean_loc[n] + np.dot(kfgain, zrsd_modens)

                reducedgain = np.dot(xptb_modens[:,n].T, evecs)*(1.-np.sqrt((nems_modens-1)/b))*(1./evals)
                reducedgain = np.dot(reducedgain, np.dot(evecs.T, YbRinv))
                xptb_loc[:,n] = xptb_loc[:,n] - np.dot(reducedgain, zptb.T).T


    if nems <= 1:
        return xmean_loc, xptb_loc, 1

    if mpi:
        comm.Bcast(xmean_loc, root=0)
        comm.Bcast(xptb_loc,  root=0)

    xmean_a = xmean_loc.copy()
    xptb_a  = np.empty_like(xptb_loc)

    if (not mpi) or (myrank == 0):
        Pa = (xptb_loc.T @ xptb_loc) / float(nems - 1)
        var_a = np.var(xptb_loc, axis=0, ddof=1)
        s_a = np.sqrt(np.mean(var_a))

        alpha_rtp = 0.6
        if s_a > 1e-12 and s_b > 0.0:
            s_target = (1.0 - alpha_rtp) * s_a + alpha_rtp * s_b
            gamma = s_target / s_a
        else:
            gamma = 1.0

        gamma = min(gamma, gamma_max)

        evals, evecs = np.linalg.eigh(Pa)
        evals = np.maximum(evals, 0.0)
        idx = np.argsort(evals)[::-1]
        evals = evals[idx]
        evecs = evecs[:, idx]

        A_sigma = np.zeros((nems, ndim), dtype=xptb_loc.dtype)

        for j in range(nem_sclfct):
            lam_j = evals[j]
            if lam_j <= 0.0:
                continue
            v_j = evecs[:, j]
            amp = gamma * np.sqrt(0.5 * (nems - 1) * lam_j)
            A_sigma[2*j,   :] =  amp * v_j
            A_sigma[2*j+1, :] = -amp * v_j

        A_sigma -= A_sigma.mean(axis=0, keepdims=True)

        xmean_a[:] = xmean_loc
        xptb_a[:, :] = A_sigma

    if mpi:
        comm.Bcast(xmean_a, root=0)
        comm.Bcast(xptb_a,  root=0)

    return xmean_a, xptb_a, 1


def mlutkf_rloc(xmean, xptb, h, h_type, obs, obs_r, obs_rloc, nem_sclfct, mpi=False, comm=None, myrank=0):
    """
    MLUTKF (R-localization only):
      - R-localization: uses obs_rloc (state-dependent R weighting)
      - NO B-localization: no modulation ensemble (no z)
      - Sigma-point reconstruction at the end
    """
    nems, ndim = xptb.shape

    # prior spread scale
    if nems > 1:
        var_b = np.var(xptb, axis=0, ddof=1)
        s_b = np.sqrt(np.mean(var_b))
    else:
        s_b = 0.0

    xmean_loc = xmean.copy()
    xptb_loc  = xptb.copy()

    nems, ndim = xptb_loc.shape
    nobs = obs.shape[-1]
    svd_calc = True

    # NO modulation ensemble
    nems_modens = nems
    xptb_modens = xptb_loc

    # Observation in ensemble space
    if h_type == 0:
        zptb = np.empty((nems, nobs), xptb_loc.dtype)
        zptb_modens = np.empty((nems_modens, nobs), xptb_loc.dtype)

        for iens in range(nems):
            zptb[iens] = np.dot(h, xptb_loc[iens])
            zptb[iens] = nonlinear_h(zptb[iens], h_type)

        zptb_modens[:, :] = zptb[:, :]

        zmean_modens = np.dot(h, xmean_loc)
        zmean_modens = nonlinear_h(zmean_modens, h_type)
        zrsd_modens = obs - zmean_modens
    else:
        x = xmean_loc + xptb_loc
        zloc = np.empty((nems, nobs), xptb_loc.dtype)

        for iens in range(nems):
            zloc[iens] = np.dot(h, x[iens])
            zloc[iens] = nonlinear_h(zloc[iens], h_type)

        zmean = zloc.mean(axis=0)
        zptb = zloc - zmean

        zptb_modens = zptb
        zmean_modens = zmean
        zrsd_modens = obs - zmean_modens

    # R-localization weights guard
    obs_rloc = np.where(obs_rloc < 1.e-13, 1.e-13, obs_rloc)

    s_floor   = 1e-10  
    gamma_max = 1.5 

    # Gain-form update (with R-loc)
    if svd_calc:
        if mpi:
            n = myrank
            xmean_send = np.array(0.0, dtype=float)
            xptb_send  = np.zeros((nems,), dtype=float)

            if n < ndim:
                Rsqrt_inv = np.sqrt(obs_rloc[n, :] / obs_r)
                YbRsqrtinv = zptb_modens * Rsqrt_inv

                u, s, v = svd(YbRsqrtinv, full_matrices=False, lapack_driver='gesvd')
                s_safe = np.where(s < s_floor, s_floor, s)

                sp = (nems_modens - 1) + (s_safe**2)
                painv = (u * (1.0 / sp)).dot(u.T)

                kfgain = np.dot(xptb_modens[:, n].T, np.dot(painv, YbRsqrtinv * Rsqrt_inv))
                xmean_loc[n] = xmean_loc[n] + np.dot(kfgain, zrsd_modens)

                reducedgain = np.dot(xptb_modens[:, n].T, u) * (1.0 - np.sqrt((nems_modens - 1) / sp))
                reducedgain = np.dot(reducedgain, (v.T / s_safe).T) * Rsqrt_inv
                xptb_loc[:, n] = xptb_loc[:, n] - np.dot(reducedgain, zptb.T).T

                xmean_send = np.array(xmean_loc[n], dtype=float)
                xptb_send[:] = xptb_loc[:, n]

            xmean_all = np.empty((ndim,), dtype=float)
            comm.Allgather(xmean_send, xmean_all)

            xptb_all_cols = np.empty((ndim, nems), dtype=float)
            comm.Allgather(xptb_send, xptb_all_cols)

            xmean_loc[:] = xmean_all
            xptb_loc[:, :] = xptb_all_cols.T
        else:
            for n in range(ndim):
                Rsqrt_inv = np.sqrt(obs_rloc[n, :] / obs_r)
                YbRsqrtinv = zptb_modens * Rsqrt_inv

                u, s, v = svd(YbRsqrtinv, full_matrices=False, lapack_driver='gesvd')
                s_safe = np.where(s < s_floor, s_floor, s) 

                sp = (nems_modens - 1) + (s_safe**2)
                painv = (u * (1.0 / sp)).dot(u.T)

                kfgain = np.dot(xptb_modens[:, n].T, np.dot(painv, YbRsqrtinv * Rsqrt_inv))
                xmean_loc[n] = xmean_loc[n] + np.dot(kfgain, zrsd_modens)

                reducedgain = np.dot(xptb_modens[:, n].T, u) * (1.0 - np.sqrt((nems_modens - 1) / sp))
                reducedgain = np.dot(reducedgain, (v.T / s_safe).T) * Rsqrt_inv 
                xptb_loc[:, n] = xptb_loc[:, n] - np.dot(reducedgain, zptb.T).T
    else:
        raise NotImplementedError("eigen-branch omitted")

    if nems <= 1:
        return xmean_loc, xptb_loc, 1

    if mpi:
        comm.Bcast(xmean_loc, root=0)
        comm.Bcast(xptb_loc,  root=0)

    # Sigma-point reconstruction
    xmean_a = xmean_loc.copy()
    xptb_a  = np.empty_like(xptb_loc)

    if (not mpi) or (myrank == 0):
        Pa = (xptb_loc.T @ xptb_loc) / float(nems - 1)
        var_a = np.var(xptb_loc, axis=0, ddof=1)
        s_a = np.sqrt(np.mean(var_a))

        alpha_rtp = 0.6
        if s_a > 1e-12 and s_b > 0.0:
            s_target = (1.0 - alpha_rtp) * s_a + alpha_rtp * s_b
            gamma = s_target / s_a
        else:
            gamma = 1.0

        gamma = min(gamma, gamma_max)

        evals, evecs = np.linalg.eigh(Pa)
        evals = np.maximum(evals, 0.0)
        idx = np.argsort(evals)[::-1]
        evals = evals[idx]
        evecs = evecs[:, idx]

        A_sigma = np.zeros((nems, ndim), dtype=xptb_loc.dtype)

        for j in range(nem_sclfct):
            lam_j = evals[j]
            if lam_j <= 0.0:
                continue
            v_j = evecs[:, j]
            amp = gamma * np.sqrt(0.5 * (nems - 1) * lam_j)
            A_sigma[2*j,   :] =  amp * v_j
            A_sigma[2*j+1, :] = -amp * v_j

        A_sigma -= A_sigma.mean(axis=0, keepdims=True)

        xmean_a[:] = xmean_loc
        xptb_a[:, :] = A_sigma

    if mpi:
        comm.Bcast(xmean_a, root=0)
        comm.Bcast(xptb_a,  root=0)

    return xmean_a, xptb_a, 1


def mlutkf_bloc(xmean, xptb, h, h_type, obs, obs_r, z, nem_sclfct, mpi=False, comm=None, myrank=0):
    """
    MLUTKF (B-localization only):
      - B-localization: modulation ensemble via z
      - NO R-localization: uniform obs_r
      - Sigma-point reconstruction at the end
    """
    nems, ndim = xptb.shape

    if nems > 1:
        var_b = np.var(xptb, axis=0, ddof=1)
        s_b = np.sqrt(np.mean(var_b))
    else:
        s_b = 0.0

    xmean_loc = xmean.copy()
    xptb_loc  = xptb.copy()

    nems, ndim = xptb_loc.shape
    nobs = obs.shape[-1]
    svd_calc = True

    if z is None:
        raise ValueError("z not specified")

    # modulation ensemble
    neig = z.shape[0]
    nems_modens = neig * nems

    xptb_modens = np.zeros((nems_modens, ndim), xptb_loc.dtype)
    iens_modens = 0
    for j in range(neig):
        for iens in range(nems):
            xptb_modens[iens_modens, :] = xptb_loc[iens, :] * z[neig - j - 1, :]
            iens_modens += 1

    xptb_modens = np.sqrt(float(nems_modens - 1) / float(nems - 1)) * xptb_modens

    # Observation in ensemble space
    if h_type == 0:
        zptb = np.empty((nems, nobs), xptb_loc.dtype)
        zptb_modens = np.empty((nems_modens, nobs), xptb_modens.dtype)

        for iens in range(nems):
            zptb[iens] = np.dot(h, xptb_loc[iens])
            zptb[iens] = nonlinear_h(zptb[iens], h_type)

        for iens in range(nems_modens):
            zptb_modens[iens] = np.dot(h, xptb_modens[iens])
            zptb_modens[iens] = nonlinear_h(zptb_modens[iens], h_type)

        zmean_modens = np.dot(h, xmean_loc)
        zmean_modens = nonlinear_h(zmean_modens, h_type)
        zrsd_modens = obs - zmean_modens
    else:
        x = xmean_loc + xptb_loc
        zloc = np.empty((nems, nobs), xptb_loc.dtype)
        for iens in range(nems):
            zloc[iens] = np.dot(h, x[iens])
            zloc[iens] = nonlinear_h(zloc[iens], h_type)

        zmean = zloc.mean(axis=0)
        zptb = zloc - zmean

        x_modens = xmean_loc + xptb_modens
        z_modens = np.empty((nems_modens, nobs), x_modens.dtype)
        for iens in range(nems_modens):
            z_modens[iens] = np.dot(h, x_modens[iens])
            z_modens[iens] = nonlinear_h(z_modens[iens], h_type)

        zmean_modens = z_modens.mean(axis=0)
        zptb_modens = z_modens - zmean_modens
        zrsd_modens = obs - zmean_modens

    s_floor   = 1e-10
    gamma_max = 1.5

    # Gain-form update (NO R-loc)
    if svd_calc:
        if mpi:
            n = myrank
            xmean_send = np.array(0.0, dtype=float)
            xptb_send  = np.zeros((nems,), dtype=float)

            if n < ndim:
                Rsqrt_inv = 1.0 / np.sqrt(obs_r)
                YbRsqrtinv = zptb_modens * Rsqrt_inv

                u, s, v = svd(YbRsqrtinv, full_matrices=False, lapack_driver='gesvd')
                s_safe = np.where(s < s_floor, s_floor, s)

                sp = (nems_modens - 1) + (s_safe**2)
                painv = (u * (1.0 / sp)).dot(u.T)

                kfgain = np.dot(xptb_modens[:, n].T, np.dot(painv, YbRsqrtinv * Rsqrt_inv))
                xmean_loc[n] = xmean_loc[n] + np.dot(kfgain, zrsd_modens)

                reducedgain = np.dot(xptb_modens[:, n].T, u) * (1.0 - np.sqrt((nems_modens - 1) / sp))
                reducedgain = np.dot(reducedgain, (v.T / s_safe).T) * Rsqrt_inv
                xptb_loc[:, n] = xptb_loc[:, n] - np.dot(reducedgain, zptb.T).T

                xmean_send = np.array(xmean_loc[n], dtype=float)
                xptb_send[:] = xptb_loc[:, n]

            xmean_all = np.empty((ndim,), dtype=float)
            comm.Allgather(xmean_send, xmean_all)

            xptb_all_cols = np.empty((ndim, nems), dtype=float)
            comm.Allgather(xptb_send, xptb_all_cols)

            xmean_loc[:] = xmean_all
            xptb_loc[:, :] = xptb_all_cols.T
        else:
            Rsqrt_inv = 1.0 / np.sqrt(obs_r)
            for n in range(ndim):
                YbRsqrtinv = zptb_modens * Rsqrt_inv

                u, s, v = svd(YbRsqrtinv, full_matrices=False, lapack_driver='gesvd')
                s_safe = np.where(s < s_floor, s_floor, s)

                sp = (nems_modens - 1) + (s_safe**2)
                painv = (u * (1.0 / sp)).dot(u.T)

                kfgain = np.dot(xptb_modens[:, n].T, np.dot(painv, YbRsqrtinv * Rsqrt_inv))
                xmean_loc[n] = xmean_loc[n] + np.dot(kfgain, zrsd_modens)

                reducedgain = np.dot(xptb_modens[:, n].T, u) * (1.0 - np.sqrt((nems_modens - 1) / sp))
                reducedgain = np.dot(reducedgain, (v.T / s_safe).T) * Rsqrt_inv
                xptb_loc[:, n] = xptb_loc[:, n] - np.dot(reducedgain, zptb.T).T
    else:
        raise NotImplementedError("eigen-branch omitted")

    if nems <= 1:
        return xmean_loc, xptb_loc, 1

    if mpi:
        comm.Bcast(xmean_loc, root=0)
        comm.Bcast(xptb_loc,  root=0)

    # Sigma-point reconstruction
    xmean_a = xmean_loc.copy()
    xptb_a  = np.empty_like(xptb_loc)

    if (not mpi) or (myrank == 0):
        Pa = (xptb_loc.T @ xptb_loc) / float(nems - 1)
        var_a = np.var(xptb_loc, axis=0, ddof=1)
        s_a = np.sqrt(np.mean(var_a))

        alpha_rtp = 0.6
        if s_a > 1e-12 and s_b > 0.0:
            s_target = (1.0 - alpha_rtp) * s_a + alpha_rtp * s_b
            gamma = s_target / s_a
        else:
            gamma = 1.0

        gamma = min(gamma, gamma_max)

        evals, evecs = np.linalg.eigh(Pa)
        evals = np.maximum(evals, 0.0)
        idx = np.argsort(evals)[::-1]
        evals = evals[idx]
        evecs = evecs[:, idx]

        A_sigma = np.zeros((nems, ndim), dtype=xptb_loc.dtype)

        for j in range(nem_sclfct):
            lam_j = evals[j]
            if lam_j <= 0.0:
                continue
            v_j = evecs[:, j]
            amp = gamma * np.sqrt(0.5 * (nems - 1) * lam_j)
            A_sigma[2*j,   :] =  amp * v_j
            A_sigma[2*j+1, :] = -amp * v_j

        A_sigma -= A_sigma.mean(axis=0, keepdims=True)

        xmean_a[:] = xmean_loc
        xptb_a[:, :] = A_sigma

    if mpi:
        comm.Bcast(xmean_a, root=0)
        comm.Bcast(xptb_a,  root=0)

    return xmean_a, xptb_a, 1
