#ifndef RootFinding_H_
#define RootFinding_H_

#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>
#include <cmath>
#include <functional>

/**
 * @brief Performs Newton-Raphson iteration to find the root of the function f.
 *
 * @param x The initial guess for x. Replaced by the value of x for which
 * f(x)=0 upon function return.
 * @param f The function to find the root for.
 * @param f_deriv The derivative of the function with respect to x.
 * @param TOL Optional. The relative tolerance to use to check for convergence.
 * @returns unsigned int The number of iterations taken to converge.
 */
template <class F1, class F2>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE unsigned int
newton_raphson(amrex::Real &x, F1 &&f, F2 &&f_deriv, amrex::Real TOL = 1e-8);

//
// Implementations
//
template <class F1, class F2>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE unsigned int
newton_raphson(amrex::Real &x, F1 &&f, F2 &&f_deriv, amrex::Real TOL)
{
    unsigned int num_iter = 0;
    amrex::Real  x_prev;
    do {
        num_iter++;
        x_prev = x;
        x      = x_prev - f(x_prev) / f_deriv(x_prev);
    } while (std::fabs(x - x_prev) >= TOL * std::fabs(x_prev));

    return num_iter;
}

#endif