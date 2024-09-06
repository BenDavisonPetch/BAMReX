/**
 * Routines for updating states given fluxes
 */
#ifndef Update_H_
#define Update_H_

#include <AMReX_GpuControl.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>

#define MAX_UPDATE_STEPS 16

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

AMREX_GPU_HOST
AMREX_FORCE_INLINE
void conservative_update(
    const amrex::Geometry &geom, const amrex::MultiFab &statein,
    const amrex::Vector<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> >
                    &fluxes,
    amrex::MultiFab &stateout1, amrex::MultiFab &stateout2,
    const amrex::Vector<amrex::Real> &dt1,
    const amrex::Vector<amrex::Real> &dt2, int N);

//! Adds src to dst using boxes nbx
AMREX_GPU_HOST
AMREX_FORCE_INLINE
void add_fluxes(
    const amrex::Array<amrex::FArrayBox *, AMREX_SPACEDIM>       &dst,
    const amrex::Array<const amrex::FArrayBox *, AMREX_SPACEDIM> &src,
    const amrex::GpuArray<amrex::Box, BL_SPACEDIM>               &nbx);

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
    BL_PROFILE("conservative_update(geom, statein, fluxes, stateout, dt)");
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
    BL_PROFILE("conservative_update(bx, statein, fx, fy, fz)");
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

    ParallelFor(bx, statein.nComp(),
                [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                {
                    out(i, j, k, n) = in(i, j, k, n)AMREX_D_TERM(
                        -dtdx * (fluxx(i + 1, j, k, n) - fluxx(i, j, k, n)),
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
    BL_PROFILE("conservative_update(geom, statein, fluxes, stateout, dt, N)");
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
    BL_PROFILE(
        "conservative_update(bx, statein, fluxes, stateout, dt, idx, N)");
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

AMREX_GPU_HOST
AMREX_FORCE_INLINE
void conservative_update(
    const amrex::Geometry &geom, const amrex::MultiFab &statein,
    const amrex::Vector<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> >
                    &fluxes,
    amrex::MultiFab &stateout1, amrex::MultiFab &stateout2,
    const amrex::Vector<amrex::Real> &dt1,
    const amrex::Vector<amrex::Real> &dt2, int N)
{
    BL_PROFILE("conservative_update(geom, statein, fluxes, stateout1, "
               "stateout2, dt1, dt2, N)");
    using namespace amrex;

    AMREX_ASSERT(statein.nComp() == stateout1.nComp());
    AMREX_ASSERT(statein.nComp() == stateout2.nComp());
    AMREX_ASSERT(fluxes.size() >= N);
    AMREX_ASSERT(dt1.size() >= N);
    AMREX_ASSERT(dt2.size() >= N);
    AMREX_ASSERT(MAX_UPDATE_STEPS >= N);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    {
        const auto &idx = geom.InvCellSizeArray();

        GpuArray<Real, MAX_UPDATE_STEPS> AMREX_D_DECL(dtdx1, dtdy1, dtdz1);
        GpuArray<Real, MAX_UPDATE_STEPS> AMREX_D_DECL(dtdx2, dtdy2, dtdz2);

        for (int iN = 0; iN < N; ++iN)
        {
            AMREX_D_TERM(dtdx1[iN] = dt1[iN] * idx[0];
                         , dtdy1[iN] = dt1[iN] * idx[1];
                         , dtdz1[iN] = dt1[iN] * idx[2];)
            AMREX_D_TERM(dtdx2[iN] = dt2[iN] * idx[0];
                         , dtdy2[iN] = dt2[iN] * idx[1];
                         , dtdz2[iN] = dt2[iN] * idx[2];)
        }

        for (MFIter mfi(statein, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const auto &bx = mfi.tilebox();

            const auto &out1 = stateout1.array(mfi);
            const auto &out2 = stateout2.array(mfi);
            const auto &in   = statein.const_array(mfi);
            GpuArray<GpuArray<Array4<const Real>, AMREX_SPACEDIM>,
                     MAX_UPDATE_STEPS>
                fluxarr;
            for (int d = 0; d < AMREX_SPACEDIM; ++d)
                for (int iN = 0; iN < N; ++iN)
                    fluxarr[iN][d] = fluxes[iN][d].const_array(mfi);

            ParallelFor(bx, statein.nComp(),
                        [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                        {
                            out1(i, j, k, n) = in(i, j, k, n);
                            out2(i, j, k, n) = in(i, j, k, n);
                            for (int iN = 0; iN < N; ++iN)
                            {
                                out1(i, j, k, n) -= AMREX_D_TERM(
                                    dtdx1[iN]
                                        * (fluxarr[iN][0](i + 1, j, k, n)
                                           - fluxarr[iN][0](i, j, k, n)),
                                    +dtdy1[iN]
                                        * (fluxarr[iN][1](i, j + 1, k, n)
                                           - fluxarr[iN][1](i, j, k, n)),
                                    +dtdz1[iN]
                                        * (fluxarr[iN][2](i, j, k + 1, n)
                                           - fluxarr[iN][2](i, j, k, n)));
                                out2(i, j, k, n) -= AMREX_D_TERM(
                                    dtdx2[iN]
                                        * (fluxarr[iN][0](i + 1, j, k, n)
                                           - fluxarr[iN][0](i, j, k, n)),
                                    +dtdy2[iN]
                                        * (fluxarr[iN][1](i, j + 1, k, n)
                                           - fluxarr[iN][1](i, j, k, n)),
                                    +dtdz2[iN]
                                        * (fluxarr[iN][2](i, j, k + 1, n)
                                           - fluxarr[iN][2](i, j, k, n)));
                            }
                        });
        }
    }
}

//! Adds src to dst using boxes nbx
AMREX_GPU_HOST
AMREX_FORCE_INLINE
void add_fluxes(
    const amrex::Array<amrex::FArrayBox *, AMREX_SPACEDIM>       &dst,
    const amrex::Array<const amrex::FArrayBox *, AMREX_SPACEDIM> &src,
    const amrex::GpuArray<amrex::Box, BL_SPACEDIM>               &nbx)
{
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
    {
        const auto &d = dst[idim]->array();
        const auto &s = src[idim]->const_array();
        ParallelFor(nbx[idim], src[idim]->nComp(),
                    [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept
                    { d(i, j, k, n) += s(i, j, k, n); });
    }
}

#endif