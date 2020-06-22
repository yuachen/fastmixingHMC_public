# sampling from nonisotropic gaussian
# fix kappa, vary d
import numpy as np
import scipy.stats

import mcmc
import sys
import time as time



# logconcave function
def f(x, mean = 0., sigma = 1.):
    # x is of dimension n * d
    return 0.5*np.sum((x - mean)**2/sigma**2, axis = 1)

# density of f up to a constant
def density_f(x, mean = 0., sigma = 1.):
    return  np.exp(-f(x, mean, sigma))

def grad_f(x, mean = 0., sigma = 1.):
    # x is of dimension n * d
    return (x - mean)/sigma**2

def main_simu(d, nb_exps=100, nb_iters=40000, sigma_max=2.0, seed=1):
    np.random.seed(123456+seed)
    error_hmc_all = np.zeros((nb_iters, 1))
    error_hmcagg_all = np.zeros((nb_iters, 1))
    error_mala_all = np.zeros((nb_iters, 1))
    error_rwmh_all = np.zeros((nb_iters, 1))

    mean = np.zeros(d)
    sigma =  np.array([1.0 + (sigma_max - 1.0)/(d-1)*i for i in range(d)])
    L = 1./sigma[0]**2
    m = 1./sigma[-1]**2
    kappa = L/m

    print("d = %d, m = %0.2f, L = %0.2f, kappa = %0.2f" %(d, m, L, kappa))

    def error_quantile(x_curr):
        q3 =  sigma[-1]*scipy.stats.norm.ppf(0.75)
        e1 = np.abs(np.percentile(x_curr[:, -1], 75) - q3)/q3
        return np.array([e1])

    init_distr = 1./np.sqrt(L)*np.random.randn(nb_exps, d)

    def grad_f_local(x):
        return grad_f(x, mean=mean, sigma=sigma)

    def f_local(x):
        return f(x, mean=mean, sigma=sigma)

    # cK is a multiplier on K
    cK = 1 
    # number of leapfrog updates
    K_hmc = np.int(np.ceil(d**0.25 * cK))
    nb_hmc_iters = np.int(nb_iters/K_hmc)+1

    error_hmc_all[:nb_hmc_iters,:], x_hmc, ac_hmc, _ = mcmc.hmc(init_distr, grad_f_local, f_local, error_quantile,
                                               stepchoice=2, kappa=kappa, L=L, L3=L, cK=cK,
                                                  nb_iters=nb_hmc_iters, nb_exps=nb_exps)  
    # aggressive HMC step-size choice by assuming L3 small
    K_hmcagg = np.int(np.ceil(d**0.125 * kappa**0.25 * cK))
    nb_hmcagg_iters = np.int(nb_iters/K_hmcagg)+1
    error_hmcagg_all[:nb_hmcagg_iters,:], x_hmcagg, ac_hmcagg, _ = mcmc.hmc(init_distr, grad_f_local, f_local, error_quantile,
                                               stepchoice=5, kappa=kappa, L=L, L3=L, cK=cK,
                                                  nb_iters=nb_hmcagg_iters, nb_exps=nb_exps)
    
    error_mala_all, x_mala, ac_mala = mcmc.mala(init_distr, grad_f_local, f_local, error_quantile,
                                           kappa=kappa, L=L, nb_iters=nb_iters, nb_exps=nb_exps)
    error_rwmh_all, x_rwmh, ac_rwmh = mcmc.rwmh(init_distr, f_local, error_quantile,
                                           kappa=kappa, L=L, nb_iters=nb_iters, nb_exps=nb_exps)

    result = {}
    result['d'] = d
    result['nb_iters'] = nb_iters
    result['nb_exps'] = nb_exps
    result['sigma_max'] = sigma_max
    result['cK'] = cK
    result['K_hmc'] = K_hmc
    result['nb_hmc_iters'] = nb_hmc_iters
    result['K_hmcagg'] = K_hmcagg
    result['nb_hmcagg_iters'] = nb_hmcagg_iters
    result['hmc'] = error_hmc_all
    result['hmcagg'] = error_hmcagg_all
    result['mala'] = error_mala_all
    result['rwmh'] = error_rwmh_all
    result['ac_hmc'] = ac_hmc
    result['ac_hmcagg'] = ac_hmcagg
    result['ac_mala'] = ac_mala
    result['ac_rwmh'] = ac_rwmh

    save_path = "/cluster/scratch/chenyua/HMC/results"
    np.save('%s/warm_gaussian_nonisotropic_d%d_cK%d_iters%d_exps%d_seed%d.npy' %(save_path, d, cK, nb_iters, nb_exps, seed), result)

if __name__ == '__main__':
    d = int(sys.argv[1])
    seed = int(sys.argv[2])
    nb_exps = int(sys.argv[3])
    main_simu(d=d, nb_exps=nb_exps, nb_iters=40000, sigma_max=2.0, seed=seed)
