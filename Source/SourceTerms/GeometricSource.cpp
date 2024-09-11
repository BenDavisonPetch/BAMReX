#include "GeometricSource.H"
#include "AmrLevelAdv.H"
#include "BCs.H"
#include "Visc/Visc_K.H"
#include <AMReX_Arena.H>
#include <AMReX_Extension.H>
#include <AMReX_MFIter.H>

using namespace amrex;

namespace details
{

//! KN = dt * f(arg)
AMREX_GPU_HOST
void visc_RK_step(Real dt, Real dr, Real drinv, Real dzinv, Real r_lo,
                  int alpha, int rot_axis, MultiFab &KN, int KNgrow,
                  const MultiFab &arg);

}

/**
 * Uses RK4 time-stepping to advance Euler equations geometric source terms for
 * methods using operator splitting.
 */
AMREX_GPU_HOST
void advance_geometric([[maybe_unused]] const amrex::Geometry &geom,
                       [[maybe_unused]] amrex::Real            dt,
                       [[maybe_unused]] int                    alpha,
                       [[maybe_unused]] int                    rot_axis,
                       [[maybe_unused]] const amrex::MultiFab &Uin,
                       [[maybe_unused]] amrex::MultiFab       &Uout)
{
#if AMREX_SPACEDIM == 1 || AMREX_SPACEDIM == 2

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    {
        for (MFIter mfi(Uout, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const auto &bx = mfi.tilebox();
#if AMREX_SPACEDIM == 1
            AMREX_ASSERT(rot_axis == 0);
            const Real dr   = geom.CellSize(0);
            const Real r_lo = geom.ProbLo(0);
            advance_geometric_1d(dt, dr, r_lo, alpha, bx, Uin[mfi], Uout[mfi]);
#elif AMREX_SPACEDIM == 2
            AMREX_ASSERT(rot_axis == 0 || rot_axis == 1);
            AMREX_ASSERT(alpha == 1); // must be cylindrical symmetry in 2D
            const Real dr   = geom.CellSize(rot_axis);
            const Real r_lo = geom.ProbLo(rot_axis);
            switch (rot_axis)
            {
            case 0:
                advance_geometric_2d<0>(dt, dr, r_lo, bx, Uin[mfi], Uout[mfi]);
                break;
            case 1:
                advance_geometric_2d<1>(dt, dr, r_lo, bx, Uin[mfi], Uout[mfi]);
                break;
            }
#endif
        }
    }

#endif
}

/**
 * Uses RK4 time-stepping to advance Navier-Stokes equations geometric source
 * terms for methods using operator splitting. Currently only supports
 * cylindrical symmetry.
 *
 * Because the viscous source terms include cell-centred gradients, we need to
 * fill boundaries between each RK step, hence this routine operates at the
 * MultiFab level rather than the FAB level
 */
AMREX_GPU_HOST
void advance_geometric_visc(const amrex::Geometry &geom, amrex::Real dt,
                            amrex::Real time, int alpha, int rot_axis,
                            const amrex::MultiFab &statein,
                            amrex::MultiFab &stateout, const bamrexBCData &bc)
{
#if AMREX_SPACEDIM == 1 || AMREX_SPACEDIM == 2
    AMREX_ASSERT(rot_axis < AMREX_SPACEDIM);
    //! Spherical symmetry not implemented
    AMREX_ASSERT(alpha == 1);
    //! 4 Ghost cells required for coarse-fine boundary when using RK4
    AMREX_ASSERT(statein.nGrow() >= 4);
    //! We use stateout as scratch space so it also needs ghost cells
    AMREX_ASSERT(stateout.nGrow() >= 4);

    constexpr int NCOMP = UInd::size;
    AMREX_ASSERT(statein.nComp() == NCOMP);
    AMREX_ASSERT(stateout.nComp() == NCOMP);

    const int  z_axis = (rot_axis == 0) ? 1 : 0;
    const Real dr     = geom.CellSize(rot_axis);
    const Real dz     = (AMREX_SPACEDIM == 2) ? geom.CellSize(z_axis) : 0;
    const Real drinv  = (Real)1 / dr;
    const Real dzinv  = (AMREX_SPACEDIM == 2) ? (Real)1 / dz : 0;
    const Real r_lo   = geom.ProbLo(rot_axis);

    //! Temporary MultiFabs
    const auto &ba = stateout.boxArray();
    const auto &dm = stateout.DistributionMap();
    MultiFab    K1(ba, dm, NCOMP, 3);
    MultiFab    K2(ba, dm, NCOMP, 2);
    MultiFab    K3(ba, dm, NCOMP, 1);
    MultiFab    K4(ba, dm, NCOMP, 0);

    //! k1 = dt * f(y_0, t_0)
    MultiFab::Copy(stateout, statein, 0, 0, NCOMP, 4);
    bc.fill_consv_boundary(geom, stateout, time);
    details::visc_RK_step(dt, dr, drinv, dzinv, r_lo, alpha, rot_axis, K1, 3,
                          stateout);
    //! k2 = dt * f(y_0 + k1/2, t_0 + dt/2)
    MultiFab::LinComb(stateout, 1, statein, 0, 0.5, K1, 0, 0, NCOMP, 3);
    bc.fill_consv_boundary(geom, stateout, time + 0.5 * dt);
    details::visc_RK_step(dt, dr, drinv, dzinv, r_lo, alpha, rot_axis, K2, 2,
                          stateout);

    //! k3 = dt * f(y_0 + k2/2, t_0 + dt/2)
    MultiFab::LinComb(stateout, 1, statein, 0, 0.5, K2, 0, 0, NCOMP, 2);
    bc.fill_consv_boundary(geom, stateout, time + 0.5 * dt);
    details::visc_RK_step(dt, dr, drinv, dzinv, r_lo, alpha, rot_axis, K3, 1,
                          stateout);

    //! k4 = dt * f(y_0 + k3, t_0 + dt)
    MultiFab::LinComb(stateout, 1, statein, 0, 1, K3, 0, 0, NCOMP, 1);
    bc.fill_consv_boundary(geom, stateout, time + dt);
    details::visc_RK_step(dt, dr, drinv, dzinv, r_lo, alpha, rot_axis, K4, 0,
                          stateout);

    //! y_new = y_0 + (k1 + 2k2 + 2k3 + k4) / 6
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    {
        for (MFIter mfi(stateout, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const auto &tbx = mfi.tilebox();
            const auto &out = stateout.array(mfi);
            const auto &in  = statein.const_array(mfi);
            const auto &k1  = K1.const_array(mfi);
            const auto &k2  = K2.const_array(mfi);
            const auto &k3  = K3.const_array(mfi);
            const auto &k4  = K4.const_array(mfi);
            ParallelFor(tbx, NCOMP,
                        [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                        {
                            out(i, j, k, n)
                                = in(i, j, k, n)
                                  + (k1(i, j, k, n) + 2 * k2(i, j, k, n)
                                     + 2 * k3(i, j, k, n) + k4(i, j, k, n))
                                        / 6;
                        });
        }
    }
#else
    amrex::ignore_unused(geom, dt, time, alpha, rot_axis, statein, stateout,
                         bc);
#endif
}

AMREX_GPU_HOST
void advance_geometric_1d(amrex::Real dt, amrex::Real dr, amrex::Real r_lo,
                          int alpha, const amrex::Box &bx,
                          const amrex::FArrayBox &statein,
                          amrex::FArrayBox       &stateout)
{
    using namespace amrex;
    constexpr int NCOMP = 3;
    AMREX_ASSERT(statein.nComp() == NCOMP);
    AMREX_ASSERT(stateout.nComp() == NCOMP);

    amrex::FArrayBox tmpfab(bx, NCOMP * 4, The_Async_Arena());

    const auto &in  = statein.const_array();
    const auto &out = stateout.array();

    const auto &k1 = tmpfab.array(0, 3);
    const auto &k2 = tmpfab.array(NCOMP, 3);
    const auto &k3 = tmpfab.array(2 * NCOMP, 3);
    const auto &k4 = tmpfab.array(3 * NCOMP, 3);

    const Real adia = AmrLevelAdv::h_parm->adiabatic;
    const Real eps  = AmrLevelAdv::h_parm->epsilon;

    // Compute K1
    ParallelFor(bx, NCOMP,
                [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                { out(i, j, k, n) = in(i, j, k, n); });

    ParallelFor(bx,
                [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    const Real  r = (i + 0.5) * dr + r_lo;
                    const auto &k1arr
                        = geometric_source(r, alpha, adia, eps, out, i);
                    AMREX_PRAGMA_SIMD
                    for (int n = 0; n < NCOMP; ++n)
                        k1(i, j, k, n) = dt * k1arr[n];
                });

    // Compute K2
    ParallelFor(bx, NCOMP,
                [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                { out(i, j, k, n) = in(i, j, k, n) + 0.5 * k1(i, j, k, n); });

    ParallelFor(bx,
                [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    const Real  r = (i + 0.5) * dr + r_lo;
                    const auto &k2arr
                        = geometric_source(r, alpha, adia, eps, out, i);
                    AMREX_PRAGMA_SIMD
                    for (int n = 0; n < NCOMP; ++n)
                        k2(i, j, k, n) = dt * k2arr[n];
                });

    // Compute K3
    ParallelFor(bx, NCOMP,
                [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                { out(i, j, k, n) = in(i, j, k, n) + 0.5 * k2(i, j, k, n); });

    ParallelFor(bx,
                [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    const Real  r = (i + 0.5) * dr + r_lo;
                    const auto &k3arr
                        = geometric_source(r, alpha, adia, eps, out, i);
                    AMREX_PRAGMA_SIMD
                    for (int n = 0; n < NCOMP; ++n)
                        k3(i, j, k, n) = dt * k3arr[n];
                });

    // Compute K4
    ParallelFor(bx, NCOMP,
                [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                { out(i, j, k, n) = in(i, j, k, n) + k3(i, j, k, n); });

    ParallelFor(bx,
                [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    const Real  r = (i + 0.5) * dr + r_lo;
                    const auto &k4arr
                        = geometric_source(r, alpha, adia, eps, out, i);
                    AMREX_PRAGMA_SIMD
                    for (int n = 0; n < NCOMP; ++n)
                        k4(i, j, k, n) = dt * k4arr[n];
                });

    // Compute Un+1
    ParallelFor(bx, NCOMP,
                [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                {
                    out(i, j, k, n) = in(i, j, k, n)
                                      + (k1(i, j, k, n) + 2 * k2(i, j, k, n)
                                         + 2 * k3(i, j, k, n) + k4(i, j, k, n))
                                            / 6;
                });
}

template <int rot_axis>
AMREX_GPU_HOST void
advance_geometric_2d(amrex::Real dt, amrex::Real dr, amrex::Real r_lo,
                     const amrex::Box &bx, const amrex::FArrayBox &statein,
                     amrex::FArrayBox &stateout)
{
    using namespace amrex;
    constexpr int NCOMP = 4;
    AMREX_ASSERT(statein.nComp() == NCOMP);
    AMREX_ASSERT(stateout.nComp() == NCOMP);

    amrex::FArrayBox tmpfab(bx, NCOMP * 4, The_Async_Arena());

    const auto &in  = statein.const_array();
    const auto &out = stateout.array();

    const auto &k1 = tmpfab.array(0, NCOMP);
    const auto &k2 = tmpfab.array(NCOMP, NCOMP);
    const auto &k3 = tmpfab.array(2 * NCOMP, NCOMP);
    const auto &k4 = tmpfab.array(3 * NCOMP, NCOMP);

    const Real adia = AmrLevelAdv::h_parm->adiabatic;
    const Real eps  = AmrLevelAdv::h_parm->epsilon;

    // Compute K1
    ParallelFor(bx, NCOMP,
                [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                { out(i, j, k, n) = in(i, j, k, n); });

    ParallelFor(bx,
                [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    const Real  r = (rot_axis == 0) ? (i + 0.5) * dr + r_lo
                                                    : (j + 0.5) * dr + r_lo;
                    const auto &k1arr
                        = geometric_source<rot_axis>(r, adia, eps, out, i, j);
                    AMREX_PRAGMA_SIMD
                    for (int n = 0; n < NCOMP; ++n)
                        k1(i, j, k, n) = dt * k1arr[n];
                });

    // Compute K2
    ParallelFor(bx, NCOMP,
                [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                { out(i, j, k, n) = in(i, j, k, n) + 0.5 * k1(i, j, k, n); });

    ParallelFor(bx,
                [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    const Real  r = (rot_axis == 0) ? (i + 0.5) * dr + r_lo
                                                    : (j + 0.5) * dr + r_lo;
                    const auto &k2arr
                        = geometric_source<rot_axis>(r, adia, eps, out, i, j);
                    AMREX_PRAGMA_SIMD
                    for (int n = 0; n < NCOMP; ++n)
                        k2(i, j, k, n) = dt * k2arr[n];
                });

    // Compute K3
    ParallelFor(bx, NCOMP,
                [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                { out(i, j, k, n) = in(i, j, k, n) + 0.5 * k2(i, j, k, n); });

    ParallelFor(bx,
                [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    const Real  r = (rot_axis == 0) ? (i + 0.5) * dr + r_lo
                                                    : (j + 0.5) * dr + r_lo;
                    const auto &k3arr
                        = geometric_source<rot_axis>(r, adia, eps, out, i, j);
                    AMREX_PRAGMA_SIMD
                    for (int n = 0; n < NCOMP; ++n)
                        k3(i, j, k, n) = dt * k3arr[n];
                });

    // Compute K4
    ParallelFor(bx, NCOMP,
                [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                { out(i, j, k, n) = in(i, j, k, n) + k3(i, j, k, n); });

    ParallelFor(bx,
                [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    const Real  r = (rot_axis == 0) ? (i + 0.5) * dr + r_lo
                                                    : (j + 0.5) * dr + r_lo;
                    const auto &k4arr
                        = geometric_source<rot_axis>(r, adia, eps, out, i, j);
                    AMREX_PRAGMA_SIMD
                    for (int n = 0; n < NCOMP; ++n)
                        k4(i, j, k, n) = dt * k4arr[n];
                });

    // Compute Un+1
    ParallelFor(bx, NCOMP,
                [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                {
                    out(i, j, k, n) = in(i, j, k, n)
                                      + (k1(i, j, k, n) + 2 * k2(i, j, k, n)
                                         + 2 * k3(i, j, k, n) + k4(i, j, k, n))
                                            / 6;
                });
}

namespace details
{

AMREX_GPU_HOST
void visc_RK_step(Real dt, Real dr, Real drinv, Real dzinv, Real r_lo,
                  int alpha, int rot_axis, MultiFab &KN, int KNgrow,
                  const MultiFab &arg)
{
    const Parm   *parm  = AmrLevelAdv::d_parm;
    constexpr int ncomp = UInd::size;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    {
        FArrayBox Qfab;
        for (MFIter mfi(arg, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const auto &kntbx = mfi.growntilebox(KNgrow);
            const auto &Qbx   = grow(kntbx, 1);
            Qfab.resize(Qbx, QInd::vsize, The_Async_Arena());
            const auto &U  = arg.const_array(mfi);
            const auto &Q  = Qfab.array();
            const auto &cQ = Qfab.const_array();
            const auto &kn = KN.array(mfi);
            ParallelFor(Qbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                        { visc_ctop(i, j, k, Q, U, *parm); });

            ParallelFor(
                kntbx,
                [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                {
                    const Real  r     = (rot_axis == 0) ? (i + 0.5) * dr + r_lo
                                                        : (j + 0.5) * dr + r_lo;
                    const auto &knarr = geometric_source_visc(
                        r, drinv, dzinv, alpha, rot_axis, *parm, cQ, i, j);
                    for (int n = 0; n < ncomp; ++n)
                    {
                        kn(i, j, k, n) = dt * knarr[n];
                    }
                });
        }
    }
}

}

// template instantiation
template AMREX_GPU_HOST void advance_geometric_2d<0>(amrex::Real, amrex::Real,
                                                     amrex::Real,
                                                     const amrex::Box &,
                                                     const amrex::FArrayBox &,
                                                     amrex::FArrayBox &);

template AMREX_GPU_HOST void advance_geometric_2d<1>(amrex::Real, amrex::Real,
                                                     amrex::Real,
                                                     const amrex::Box &,
                                                     const amrex::FArrayBox &,
                                                     amrex::FArrayBox &);