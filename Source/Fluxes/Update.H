/**
 * Routines for updating states given fluxes
 */
#ifndef Update_H_
#define Update_H_

#include <AMReX_GpuControl.H>
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

//! Applies multiple sets of fluxes with differing dt values. Uses the first \c
//! N values in each vector
AMREX_GPU_HOST
AMREX_FORCE_INLINE
void conservative_update(
    const amrex::Geometry &geom, const amrex::MultiFab &statein,
    const amrex::Vector<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> >
                    &fluxes,
    amrex::MultiFab &stateout, amrex::Vector<amrex::Real> dt, int N);

//! Applies multiple sets of fluxes with differing dt values. Uses the first \c
//! N values in each vector
AMREX_GPU_HOST
AMREX_FORCE_INLINE
void conservative_update(
    const amrex::Box &bx, const amrex::Array4<const amrex::Real> &statein,
    const amrex::Vector<amrex::Array<amrex::Array4<const amrex::Real>,
                                     AMREX_SPACEDIM> > &fluxes,
    const amrex::Array4<amrex::Real> &stateout, amrex::Vector<amrex::Real> &dt,
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> &idx, int N);

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

AMREX_GPU_HOST
AMREX_FORCE_INLINE
void conservative_update(
    const amrex::Geometry &geom, const amrex::MultiFab &statein,
    const amrex::Vector<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> >
                    &fluxes,
    amrex::MultiFab &stateout, amrex::Vector<amrex::Real> dt, int N)
{
    using namespace amrex;

    AMREX_ASSERT(fluxes.size() >= N);
    AMREX_ASSERT(dt.size() >= N);

    const auto &idx = geom.InvCellSizeArray();

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    {
        Vector<Array<Array4<const Real>, AMREX_SPACEDIM> > arrs(N);
        for (MFIter mfi(stateout, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const auto &bx = mfi.tilebox();
            for (int i = 0; i < N; ++i)
                for (int d = 0; d < AMREX_SPACEDIM; ++d)
                    arrs[i][d] = fluxes[i][d].const_array(mfi);
            conservative_update(bx, statein.const_array(mfi), arrs,
                                stateout.array(mfi), dt, idx, N);
        }
    }
}

AMREX_GPU_HOST
AMREX_FORCE_INLINE
void conservative_update(
    const amrex::Box &bx, const amrex::Array4<const amrex::Real> &statein,
    const amrex::Vector<amrex::Array<amrex::Array4<const amrex::Real>,
                                     AMREX_SPACEDIM> > &fluxes,
    const amrex::Array4<amrex::Real> &stateout, amrex::Vector<amrex::Real> &dt,
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> &idx, int N)
{
    using namespace amrex;

    AMREX_ASSERT(statein.nComp() == stateout.nComp());
    AMREX_ASSERT(fluxes.size() >= N);
    AMREX_ASSERT(dt.size() >= N);

    AMREX_D_TERM(Vector<Real> dtdx(N);, Vector<Real> dtdy(N);
                 , Vector<Real>                      dtdz(N);)
    for (int iN = 0; iN < N; ++iN)
    {
        AMREX_D_TERM(dtdx[iN] = dt[iN] * idx[0];, dtdy[iN] = dt[iN] * idx[1];
                     , dtdz[iN]                            = dt[iN] * idx[2];)
    }

    ParallelFor(bx, statein.nComp(),
                [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                {
                    stateout(i, j, k, n) = statein(i, j, k, n);
                    for (int iN = 0; iN < N; ++iN)
                    {
                        stateout(i, j, k, n) -= AMREX_D_TERM(
                            dtdx[iN]
                                * (fluxes[iN][0](i + 1, j, k, n)
                                   - fluxes[iN][0](i, j, k, n)),
                            +dtdy[iN]
                                * (fluxes[iN][1](i, j + 1, k, n)
                                   - fluxes[iN][1](i, j, k, n)),
                            +dtdz[iN]
                                * (fluxes[iN][2](i, j, k + 1, n)
                                   - fluxes[iN][2](i, j, k, n)));
                    }
                });
}

#endif