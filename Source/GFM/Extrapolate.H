#ifndef Extrapolate_H_
#define Extrapolate_H_

#include "GFM/FastSweep.H"
#include "GFM/GFMFlag.H"
#include <AMReX_Array4.H>

/**
 * grads should be zero indexed, such that grads[0] corresponds to normal
 * gradient of data[startcomp]. If grads is nullptr grads are assumed to be 0
 */
AMREX_GPU_HOST
void extrapolate_from_boundary(const amrex::Geometry            &geom,
                               amrex::MultiFab                  &data,
                               const amrex::MultiFab            &levelset,
                               const amrex::iMultiFab           &gfm_flags,
                               const amrex::Vector<amrex::Real> &grads,
                               int startcomp, int ncomp);

/**
 * @brief Extrapolates the ghost cells (that should already have been
 * filled) from the boundary into the ghost region
 *
 * grads should be zero indexed, such that grads[0] corresponds to normal
 * gradient of data[startcomp]. If grads is nullptr grads are assumed to be 0
 */
AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE void extrapolate_from_boundary(
    const amrex::Box &bx, const amrex::Array4<amrex::Real> &data,
    const amrex::Array4<const amrex::Real>             &levelset,
    const amrex::Array4<const int>                     &flag,
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> &dx,
    const amrex::Real *grads, int startcomp, int ncomp);

/**
 * grads should be zero indexed, such that grads[0] corresponds to normal
 * gradient of data[startcomp]. If grads is nullptr grads are assumed to be 0
 */
AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE void
compute_extrapolation(int i, int j, int k,
                      const amrex::Array4<const amrex::Real>             &ls,
                      const amrex::Array4<amrex::Real>                   &data,
                      const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> &dx,
                      const amrex::Real *grads, int startcomp, int ncomp);

//
// Implementations
//
AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE void extrapolate_from_boundary(
    const amrex::Box &bx, const amrex::Array4<amrex::Real> &data,
    const amrex::Array4<const amrex::Real>             &levelset,
    const amrex::Array4<const int>                     &flag,
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> &dx,
    const amrex::Real *grads, int startcomp, int ncomp)
{
    using namespace amrex;

    fast_sweep(bx, flag,
               [&](int i, int j, int k)
               {
                   if (!is_ghost(flag(i, j, k)))
                       return;
                   compute_extrapolation(i, j, k, levelset, data, dx, grads,
                                         startcomp, ncomp);
               });

    fast_sweep(bx, flag,
               [&](int i, int j, int k)
               {
                   if (!is_ghost(flag(i, j, k)))
                       return;
                   compute_extrapolation(i, j, k, levelset, data, dx, grads,
                                         startcomp, ncomp);
               });
}

AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE void
compute_extrapolation(int i, int j, int k,
                      const amrex::Array4<const amrex::Real>             &ls,
                      const amrex::Array4<amrex::Real>                   &data,
                      const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> &dx,
                      const amrex::Real *grads, int startcomp, int ncomp)
{
    using namespace amrex;
    RealVect ndx(AMREX_D_DECL(ls(i, j, k, 1) / dx[0], ls(i, j, k, 2) / dx[1],
                              ls(i, j, k, 3) / dx[2]));
    // for code below ndx should always point in direction of increasing |ls|
    // sign conventions are very weird bc of AMReX's EB thing
    const Real signflip = (ls(i, j, k) >= 0) ? -1 : 1;
    ndx *= signflip;
    IntVect ijk_x{ AMREX_D_DECL(ndx[0] < 0 ? i + 1 : i - 1, j, k) };
#if AMREX_SPACEDIM >= 2
    IntVect ijk_y{ AMREX_D_DECL(i, ndx[1] < 0 ? j + 1 : j - 1, k) };
#endif
#if AMREX_SPACEDIM == 3
    IntVect ijk_z{ AMREX_D_DECL(i, j, ndx[2] < 0 ? k + 1 : k - 1) };
#endif
    for (int n = startcomp; n < startcomp + ncomp; ++n)
    {
        // grads given as n.grad(Q) = g for n pointing outwards from rigid body
        const Real        g = (grads) ? grads[n - startcomp] * signflip : 0;
        const long double numerator = AMREX_D_TERM(
            g + fabs(ndx[0]) * data(ijk_x, n), +fabs(ndx[1]) * data(ijk_y, n),
            +fabs(ndx[2]) * data(ijk_z, n));
        const long double denominator
            = AMREX_D_TERM(fabs(ndx[0]), +fabs(ndx[1]), +fabs(ndx[2]));
        data(i, j, k, n) = numerator / denominator;
    }
}

#endif