#ifndef GeometricSource_H_
#define GeometricSource_H_

#include <AMReX_AmrLevel.H>
#include <AMReX_Array4.H>
#include <AMReX_FArrayBox.H>

AMREX_GPU_HOST
void advance_geometric(amrex::Real dt, amrex::Real dr, amrex::Real r_lo,
                       int alpha, const amrex::Box &bx,
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

#endif