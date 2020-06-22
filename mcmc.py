# mcmc package that contains random walk,
# Unadjusted Langevin Dynamics and Metropolis Adjusted Langevin Dynamics
# hamiltonian monte carlo
import numpy as np


def neglog_density_normal(x, mean=0., sigma=1.):
    """
    Negative log Density of the normal distribution up to constant
    Args:
        x (n * d): location, can be high dimension
        mean (d): mean
    Returns:
        density value at x
    """
    return np.sum((x - mean)**2, axis=1) / 2 / sigma**2

def ula(x_init, grad_f, error_metric, epsilon=1.0, kappa=1.0, L=1.0, dpower=1.0,
        nb_iters=200, nb_exps=10):
    """
    Unadjusted Langevin Dynamics for sampling

    Args:
        x_init (nb_exps * d):  the intial distribution, nb_exps * d;
        grad_f (fun):  compute the gradient of the target f function,
            taking current distribution as argument and returning the gradient
        error_metric (fun): compute the error metric
        epsilon (float): the error threshold for ULA,
            used to determine stepsize
        L (float): smoothness constant
        nb_iters (int): number of iterations
        nb_exps (int): number of sample points at each iteration

    Returns:
        error_all (nb_iters): array of all errors
        x_curr: the final sample
    """
    _, d = x_init.shape
    x_curr = x_init.copy()
    # set up the step sizes
    h_ula = 1.0 * epsilon**2 / d**dpower / kappa / L
    nh_ula = np.sqrt(2 * h_ula)

    error_1 = error_metric(x_curr)
    error_all = np.zeros((nb_iters, error_1.shape[0]))
    error_all[0] = error_1

    for i in range(nb_iters - 1):
        x_curr = x_curr - h_ula * grad_f(x_curr) \
            + nh_ula * np.random.randn(nb_exps, d)

        error_all[i + 1] = error_metric(x_curr)

    return error_all, x_curr


def mala(x_init, grad_f, f, error_metric, kappa=1.0, L=1.0, dpower=1.0, nb_iters=200, nb_exps=10):
    """
    Metropolis Adjusted Langevin Dynamics
    for sampling

    Args:
        x_init (nb_exps * d):  the intial distribution, nb_exps * d;
        grad_f (fun):  compute the gradient of the target f function,
            taking current distribution as argument and returning the gradient
        f (fun):  compute the negative log density of the target function up to constant,
        error_metric (fun): compute the error metric
        L (float): smoothness constant
        nb_iters (int): number of iterations
        nb_exps (int): number of sample points at each iteration

    Returns:
        error_all (nb_iters): array of all errors
        x_curr: the final sample
        accept_rate_all: accept rate along the samples
    """

    _, d = x_init.shape
    x_curr = x_init.copy()
    # set up the step sizes
    h_mala = 1. / L / np.maximum(d**dpower, np.sqrt(d**dpower * kappa))
    nh_mala = np.sqrt(2 * h_mala)

    error_1 = error_metric(x_curr)
    error_all = np.zeros((nb_iters, error_1.shape[0]))
    error_all[0] = error_1
    accept_rate_all = np.zeros(nb_iters)
    accept_rate_all[0] = 1.0

    for i in range(nb_iters - 1):
        proposal = x_curr - h_mala * grad_f(x_curr) \
            + nh_mala * np.random.randn(nb_exps, d)

        log_ratio = - f(proposal) \
            - neglog_density_normal(x=x_curr,
                                 mean=proposal - h_mala * grad_f(proposal),
                                 sigma=nh_mala)
        log_ratio -= - f(x_curr) \
            -  neglog_density_normal(x=proposal,
                                  mean=x_curr - h_mala * grad_f(x_curr),
                                  sigma=nh_mala)

        ratio = np.exp(log_ratio)
        # Metropolis Hastings step
        ratio = np.minimum(1., ratio)
        a = np.random.rand(nb_exps)
        index_forward = np.where(a <= ratio)[0]
        accept_rate_all[i+1] = len(index_forward)/float(nb_exps)

        x_curr[index_forward, ] = proposal[index_forward, ]

        error_all[i + 1] = error_metric(x_curr)

    return error_all, x_curr, accept_rate_all


def rwmh(x_init, f, error_metric, kappa=1.0, L=1.0, dpower=1.0, nb_iters=200, nb_exps=10):
    """
    Random walk with Metropolis Hastings
    for sampling

    Args:
        x_init (nb_exps * d):  the intial distribution, nb_exps * d;
        f (fun):  compute the negative log density of the target function up to constant,
        error_metric (fun): compute the error metric
        L (float): smoothness constant
        nb_iters (int): number of iterations
        nb_exps (int): number of sample points at each iteration

    Returns:
        error_all (nb_iters): array of all errors
        x_curr: the final sample
        accept_rate_all: accept rate along the samples
    """
    _, d = x_init.shape
    x_curr = x_init.copy()
    # set up the step sizes
    h_rwmh = 1. / d**dpower / kappa / L
    nh_rwmh = np.sqrt(2 * h_rwmh)

    error_1 = error_metric(x_curr)
    error_all = np.zeros((nb_iters, error_1.shape[0]))
    error_all[0] = error_1
    accept_rate_all = np.zeros(nb_iters)
    accept_rate_all[0] = 1.0

    for i in range(nb_iters - 1):
        proposal = x_curr + nh_rwmh * np.random.randn(nb_exps, d)

        log_ratio = - f(proposal) \
            - neglog_density_normal(x=x_curr,
                                    mean=proposal,
                                    sigma=nh_rwmh)
        log_ratio -= - f(x_curr) \
            - neglog_density_normal(x=proposal,
                                    mean=x_curr,
                                    sigma=nh_rwmh)

        ratio = np.exp(log_ratio)
        # Metropolis Hastings step
        ratio = np.minimum(1., ratio)
        a = np.random.rand(nb_exps)
        index_forward = np.where(a <= ratio)[0]
        accept_rate_all[i+1] = len(index_forward)/float(nb_exps)

        x_curr[index_forward, ] = proposal[index_forward, ]

        error_all[i + 1] = error_metric(x_curr)

    return error_all, x_curr, accept_rate_all


def hmc(x_init, grad_f, f, error_metric, stepchoice=2, kappa=1.0, L=1.0, L3=1.0, cK = 1, nb_iters=200, nb_exps=10):
    """
    Hamiltonian Monte Carlo with leapfrog integrator
    for sampling

    Args:
        x_init (nb_exps * d):  the intial distribution, nb_exps * d;
        grad_f (fun):  compute the gradient of the target f function,
            taking current distribution as argument and returning the gradient
        f (fun):  compute the negative log density of the target function up to constant,
        error_metric (fun): compute the error metric
        L (float): smoothness constant
        L3 (float): third order smoothness constant
        cK (float): constant multiplier on K, to make the integration path longer
        nb_iters (int): number of iterations
        nb_exps (int): number of sample points at each iteration
        stepchoice (int): specify the leapfrog steps and step-size choice 

    Returns:
        error_all (nb_iters): array of all errors
        x_curr: the final sample
        accept_rate_all: accept rate along the samples
    """
    _, d = x_init.shape
    x_curr = x_init.copy()
    # set up the step sizes
    if stepchoice == 4:
        # feasible start, strongly logconcave
        # number of leapfrog updates
        K = int(1 * cK)
        # step size
        eta = np.sqrt(1.0/L/d**0.5/kappa**0.5) / cK
    elif stepchoice == 3:
        # feasible start, strongly logconcave
        # kappa < d
        # number of leapfrog updates
        K = np.int(np.ceil(d**0.75/kappa**0.75 * cK))
        # step size
        eta = np.sqrt(1.0/L/d**1.5*kappa**0.5) /cK
    elif stepchoice == 2:
        # feasible start, strongly logconcave
        # kappa < d**(2/3)
        K = np.int(np.ceil(d**0.25 * cK))
        # step size
        eta = np.sqrt(1.0/L/d**(7.0/6) / cK)
    elif stepchoice == 1:
        # feasible start, strongly logconcave
        # kappa < d**(1/3)
        K = np.int(np.ceil(kappa**0.75 * cK))
        # step size
        eta = np.sqrt(1.0/L/d/kappa**0.5 / cK)
    elif stepchoice == 5:
        # feasible start, strongly logconcave
        # aggressive step-size choice, assume very small L3
        K = np.int(np.ceil(d**0.125 * kappa**0.25 * cK))
        # step size
        eta = np.sqrt(1.0/L/d**0.75/kappa**0.5 / cK)


    error_1 = error_metric(x_curr)
    error_all = np.zeros((nb_iters, error_1.shape[0]))
    error_all[0] = error_1
    accept_rate_all = np.zeros(nb_iters)
    accept_rate_all[0] = 1.0

    for i in range(nb_iters - 1):
        p0 = np.random.randn(nb_exps, d)
        p = p0
        q = x_curr
        for j in range(K):
            p = p - eta/2*grad_f(q)
            q = q + eta*p
            p = p - eta/2*grad_f(q)

        log_ratio = - f(q) - np.sum(p*p, 1)/2
        log_ratio -= - f(x_curr) - np.sum(p0*p0, 1)/2

        ratio = np.exp(log_ratio)
        # Metropolis Hastings step
        ratio = np.minimum(1., ratio)
        a = np.random.rand(nb_exps)
        index_forward = np.where(a <= ratio)[0]
        accept_rate_all[i+1] = len(index_forward)/float(nb_exps)

        x_curr[index_forward, ] = q[index_forward, ]

        error_all[i + 1] = error_metric(x_curr)

    return error_all, x_curr, accept_rate_all, K





