#ifndef GeometricSource_H_
#define GeometricSource_H_

#include <AMReX_AmrLevel.H>
#include <AMReX_Array4.H>
#include <AMReX_FArrayBox.H>

//! Convenience function, will call the correct routine below.
//! In 2D, rot_axis = 0 -> x lo, = 1 -> y lo
AMREX_GPU_HOST
void advance_geometric(const amrex::Geometry &geom, amrex::Real dt, int alpha,
                       int rot_axis, const amrex::MultiFab &statein,
                       amrex::MultiFab &stateout);

AMREX_GPU_HOST
void advance_geometric_1d(amrex::Real dt, amrex::Real dr, amrex::Real r_lo,
                          int alpha, const amrex::Box &bx,
                          const amrex::FArrayBox &statein,
                          amrex::FArrayBox       &stateout);

//! rot_axis = 0 -> x lo, = 1 -> y lo
template <int rot_axis>
AMREX_GPU_HOST void advance_geometric_2d(amrex::Real dt, amrex::Real dr,
                                         amrex::Real r_lo, int alpha,
                                         const amrex::Box       &bx,
                                         const amrex::FArrayBox &statein,
                                         amrex::FArrayBox       &stateout);

//! Geometric source term for the Euler equations
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
amrex::GpuArray<amrex::Real, 3>
geometric_source(amrex::Real r, int alpha, amrex::Real adia, amrex::Real eps,
                 const amrex::Array4<const amrex::Real> &U, int i)
{
    using namespace amrex;
    AMREX_ASSERT(U.nComp() == 3);
    GpuArray<amrex::Real, 3> src;
    const Real               prefactor = -alpha / r;
    const Real               u         = U(i, 0, 0, 1) / U(i, 0, 0, 0);
    src[0]                             = prefactor * U(i, 0, 0, 1); // rho u
    src[1] = prefactor * U(i, 0, 0, 1) * u;                         // rho u^2
    src[2] = prefactor * u
             * (adia * U(i, 0, 0, 2)
                - (adia - 1) * 0.5 * eps * U(i, 0, 0, 1) * u); // (E + p)u

    return src;
}

//! Geometric source term for the Euler equations in 2D
//! \tparam axis_loc = 0 for lo x, 1 for lo y
template <int axis_loc>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::GpuArray<amrex::Real, 4>
geometric_source(amrex::Real r, int alpha, amrex::Real adia, amrex::Real eps,
                 const amrex::Array4<const amrex::Real> &U, int i, int j)
{
    using namespace amrex;
    AMREX_ASSERT(U.nComp() == 4);
    GpuArray<amrex::Real, 4> src;
    const Real               prefactor = -alpha / r;
    constexpr int            u_r_idx   = (axis_loc == 0) ? 1 : 2;
    const Real               u_r       = U(i, j, 0, u_r_idx) / U(i, j, 0, 0);
    src[0] = prefactor * U(i, j, 0, u_r_idx); // rho u_r
    src[1] = prefactor * U(i, j, 0, 1) * u_r; // rho u_r u_r
    src[2] = prefactor * U(i, j, 0, 2) * u_r; // rho u_z u_r
    src[3] = prefactor * u_r
             * (adia * U(i, j, 0, 3)
                - (adia - 1) * 0.5 * eps
                      * (U(i, j, 0, 1) * U(i, j, 0, 1)
                         + U(i, j, 0, 2) * U(i, j, 0, 2))
                      / U(i, j, 0, 0)); // (E + p)u_r

    return src;
}

#endif