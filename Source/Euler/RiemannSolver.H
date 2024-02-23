#ifndef RiemannSolver_H_
#define RiemannSolver_H_

#include "Euler/Euler.H"
#include "RootFinding.H"
#include <AMReX_Extension.H>
#include <AMReX_GpuQualifiers.H>

/**
 * @brief Uses the Newton-Raphson root finding method to find the intermediate
 * pressure for the Euler equation Riemann problem with two ideal gases. Uses
 * the initial guess from a linearised solution based on primitive variables
 * discussed in Toro's book Riemann Solvers and Numerical Methods for Fluid
 * Dynamics
 *
 * @param dir The direction the interface is normal to
 * @param primv_L The primitive values of the left state
 * @param primv_R The primitive values of the right state
 * @param adiabatic_L The adiabatic constant of the left state
 * @param adiabatic_R The adiabatic constant of the right state
 */
AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::Real euler_RP_interm_p(
    const int dir, const amrex::GpuArray<amrex::Real, EULER_NCOMP> &primv_L,
    const amrex::GpuArray<amrex::Real, EULER_NCOMP> &primv_R,
    const amrex::Real adiabatic_L, const amrex::Real adiabatic_R);

AMREX_GPU_HOST
void compute_exact_RP_solution(
    const int dir, const amrex::Real time, const amrex::Box &bx,
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> &dx_arr,
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> &prob_lo,
    const amrex::Real                                   initial_position,
    const amrex::GpuArray<amrex::Real, EULER_NCOMP> &primv_L,
    const amrex::GpuArray<amrex::Real, EULER_NCOMP> &primv_R,
    const amrex::Real adiabatic_L, const amrex::Real adiabatic_R,
    const amrex::Array4<amrex::Real> &dest);

//
// Implementations
//
AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::Real
                                    euler_RP_interm_p(const int                                           dir,
                                                      const amrex::GpuArray<amrex::Real, EULER_NCOMP> &primv_L,
                                                      const amrex::GpuArray<amrex::Real, EULER_NCOMP> &primv_R,
                                                      const amrex::Real adiabatic_L, const amrex::Real adiabatic_R)
{
    // Calculate initial guess for pressure
    amrex::Real c_L = sound_speed(primv_L, adiabatic_L);
    amrex::Real c_R = sound_speed(primv_R, adiabatic_R);
    amrex::Real pressure
        = 0.5 * (primv_L[1 + AMREX_SPACEDIM] + primv_R[1 + AMREX_SPACEDIM])
          - 0.125 * (primv_R[1 + dir] - primv_L[1 + dir])
                * (primv_L[0] + primv_R[0]) * (c_L + c_R);
    amrex::Real TOL = 1e-6;
    pressure        = std::max(TOL, pressure);

    // Use NR method to find root
    std::function<amrex::Real(amrex::Real)> f = [=](amrex::Real p)
    {
        return euler_riemann_function(dir, p, primv_L, primv_R, adiabatic_L,
                                      adiabatic_R);
    };
    std::function<amrex::Real(amrex::Real)> f_deriv = [=](amrex::Real p)
    {
        return euler_riemann_function_deriv(dir, p, primv_L, primv_R,
                                            adiabatic_L, adiabatic_R);
    };
    newton_raphson(pressure, f, f_deriv);
    return pressure;
}

#endif