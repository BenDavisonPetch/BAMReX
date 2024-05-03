/**
 * Routines for updating states given fluxes
 */
#ifndef Update_H_
#define Update_H_

#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>

//! Dimensionally unsplit
AMREX_GPU_HOST
AMREX_FORCE_INLINE
void conservative_update(
    const amrex::Geometry &geom, const amrex::MultiFab &statein,
    const amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> &fluxes,
    amrex::MultiFab &stateout, amrex::Real dt);

//! Dimensionally unsplit
AMREX_GPU_HOST
AMREX_FORCE_INLINE
void conservative_update(
    const amrex::Box &bx, const amrex::FArrayBox &statein,
    AMREX_D_DECL(const amrex::FArrayBox &fx, const amrex::FArrayBox &fy,
                 const amrex::FArrayBox &fz),
    amrex::FArrayBox &stateout, amrex::Real dt,
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> &idx);

//
// Implementations
//

AMREX_GPU_HOST
AMREX_FORCE_INLINE
void conservative_update(
    const amrex::Geometry &geom, const amrex::MultiFab &statein,
    const amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> &fluxes,
    amrex::MultiFab &stateout, amrex::Real dt)
{
    using namespace amrex;
    const auto &idx = geom.InvCellSizeArray();
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    {
        for (MFIter mfi(stateout, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const auto &bx = mfi.tilebox();
            conservative_update(
                bx, statein[mfi],
                AMREX_D_DECL(fluxes[0][mfi], fluxes[1][mfi], fluxes[2][mfi]),
                stateout[mfi], dt, idx);
        }
    }
}

AMREX_GPU_HOST
AMREX_FORCE_INLINE
void conservative_update(
    const amrex::Box &bx, const amrex::FArrayBox &statein,
    AMREX_D_DECL(const amrex::FArrayBox &fx, const amrex::FArrayBox &fy,
                 const amrex::FArrayBox &fz),
    amrex::FArrayBox &stateout, amrex::Real dt,
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> &idx)
{
    AMREX_ASSERT(statein.nComp() == stateout.nComp());
    AMREX_ASSERT(fx.nComp() == statein.nComp());

    using namespace amrex;
    AMREX_D_TERM(const Real dtdx = dt * idx[0];, const Real dtdy = dt * idx[1];
                 , const Real dtdz = dt * idx[2];)

    const auto &in  = statein.const_array();
    const auto &out = stateout.array();
    AMREX_D_TERM(const auto &fluxx = fx.const_array();
                 , const auto &fluxy = fy.const_array();
                 , const auto &fluxz = fz.const_array();)

    ParallelFor(
        bx, statein.nComp(),
        [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
        {
            out(i, j, k, n)
                = in(i, j, k, n)
                  - AMREX_D_TERM(
                      dtdx * (fluxx(i + 1, j, k, n) - fluxx(i, j, k, n)),
                      -dtdy * (fluxy(i, j + 1, k, n) - fluxy(i, j, k, n)),
                      -dtdz * (fluxz(i, j, k + 1, n) - fluxz(i, j, k, n)));
        });
}

#endif