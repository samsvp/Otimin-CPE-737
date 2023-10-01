from numpy import (atleast_1d, eye, argmin, zeros, shape, squeeze,
                   asarray, sqrt, Inf)
import numpy as np
from scipy.optimize._differentiable_functions import ScalarFunction, FD_METHODS

from scipy.optimize import _minpack2 as minpack2


_epsilon = sqrt(np.finfo(float).eps)


def scalar_search_wolfe1(phi, derphi, phi0=None, old_phi0=None, derphi0=None,
                         c1=1e-4, c2=0.9,
                         amax=50, amin=1e-8, xtol=1e-14):

    if old_phi0 is not None and derphi0 != 0:
        alpha1 = min(1.0, 1.01*2*(phi0 - old_phi0)/derphi0)
        if alpha1 < 0:
            alpha1 = 1.0
    else:
        alpha1 = 1.0

    phi1 = phi0
    derphi1 = derphi0
    isave = np.zeros((2,), np.intc)
    dsave = np.zeros((13,), float)
    task = b'START'

    maxiter = 100
    for i in range(maxiter):
        stp, phi1, derphi1, task = minpack2.dcsrch(alpha1, phi1, derphi1,
                                                   c1, c2, xtol, task,
                                                   amin, amax, isave, dsave)
        if task[:2] == b'FG':
            alpha1 = stp
            phi1 = phi(stp)
            derphi1 = derphi(stp)
        else:
            break
    else:
        # maxiter reached, the line search did not converge
        stp = None

    return stp, phi1, phi0


def line_search_wolfe1(f, fprime, xk, pk, gfk=None,
                       old_fval=None, old_old_fval=None,
                       args=(), c1=1e-4, c2=0.9, amax=50, amin=1e-8,
                       xtol=1e-14):
    if gfk is None:
        gfk = fprime(xk, *args)

    gval = [gfk]
    gc = [0]
    fc = [0]

    def phi(s):
        fc[0] += 1
        return f(xk + s*pk, *args)

    def derphi(s):
        gval[0] = fprime(xk + s*pk, *args)
        gc[0] += 1
        return np.dot(gval[0], pk)

    derphi0 = np.dot(gfk, pk)

    stp, fval, old_fval = scalar_search_wolfe1(
            phi, derphi, old_fval, old_old_fval, derphi0,
            c1=c1, c2=c2, amax=amax, amin=amin, xtol=xtol)

    return stp, fc[0], gc[0], fval, old_fval, gval[0]

    
def vecnorm(x):
    return np.sum(np.abs(x)**2, axis=0)**(1.0 / 2)

def _prepare_scalar_function(fun, x0, jac=None, args=(),
                             epsilon=None, finite_diff_rel_step=None,
                             hess=None):
    grad = '2-point'
    epsilon = epsilon

    if hess is None:
        def hess(x, *args):
            return None
        
    bounds = (-np.inf, np.inf)

    sf = ScalarFunction(fun, x0, args, grad, hess,
                        finite_diff_rel_step, bounds, epsilon=epsilon)

    return sf


def minimize_cg(fun, x0, args=(), jac=None, callback=None,
                 gtol=1e-5, norm=Inf, eps=_epsilon, maxiter=None,
                 finite_diff_rel_step=None):
    x0 = asarray(x0).flatten()
    if maxiter is None:
        maxiter = len(x0) * 200

    sf = _prepare_scalar_function(fun, x0, jac=jac, args=args, epsilon=eps,
                                  finite_diff_rel_step=finite_diff_rel_step)

    f = sf.fun
    myfprime = sf.grad

    old_fval = f(x0)
    gfk = myfprime(x0)

    k = 0
    xk = x0
    # Sets the initial step guess to dx ~ 1
    old_old_fval = old_fval + np.linalg.norm(gfk) / 2

    warnflag = 0
    pk = -gfk
    gnorm = vecnorm(gfk)

    sigma_3 = 0.01

    while (gnorm > gtol) and (k < maxiter):
        deltak = np.dot(gfk, gfk)

        cached_step = [None]

        def polak_ribiere_powell_step(alpha, gfkp1=None):
            xkp1 = xk + alpha * pk
            if gfkp1 is None:
                gfkp1 = myfprime(xkp1)
            yk = gfkp1 - gfk
            beta_k = max(0, np.dot(yk, gfkp1) / deltak)
            pkp1 = -gfkp1 + beta_k * pk
            gnorm = vecnorm(gfkp1)
            return (alpha, xkp1, pkp1, gfkp1, gnorm)


        alpha_k, fc, gc, old_fval, old_old_fval, gfkp1 = \
                     line_search_wolfe1(f, myfprime, xk, pk, gfk, old_fval,
                                          old_old_fval, c2=0.4, amin=1e-100, amax=1e100)
        if alpha_k == None:
            break

        # Reuse already computed results if possible
        if alpha_k == cached_step[0]:
            alpha_k, xk, pk, gfk, gnorm = cached_step
        else:
            alpha_k, xk, pk, gfk, gnorm = polak_ribiere_powell_step(alpha_k, gfkp1)

        k += 1
            
    return xk
