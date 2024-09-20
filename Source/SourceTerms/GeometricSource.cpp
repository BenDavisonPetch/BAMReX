#include "GeometricSource.H"
#include "AmrLevelAdv.H"
#include "BoundaryConditions/BCs.H"
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
 * methods using operator splitting. We can do everything in one loop
 * because the Euler geometric source terms don't require any gradient
 * calculations
 */
AMREX_GPU_HOST
void advance_geometric(const amrex::Geometry &geom, amrex::Real dt, int alpha,
                       int rot_axis, const amrex::MultiFab &statein,
                       amrex::MultiFab &stateout)
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

    const Real dr   = geom.CellSize(rot_axis);
    const Real r_lo = geom.ProbLo(rot_axis);

    const Parm *parm = AmrLevelAdv::d_parm;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    {
        for (MFIter mfi(stateout, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const auto &bx = mfi.tilebox();

            const auto &in  = statein.const_array(mfi);
            const auto &out = stateout.array(mfi);

            ParallelFor(
                bx,
                [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                {
                    const Real r = (rot_axis == 0) ? (i + 0.5) * dr + r_lo
                                                   : (j + 0.5) * dr + r_lo;
                    //! k1 = f(y_0, t_0)
                    for (int n = 0; n < NCOMP; ++n)
                        out(i, j, k, n) = in(i, j, k, n);
                    const auto &k1 = geometric_source(r, alpha, rot_axis,
                                                      *parm, out, i, j);

                    //! k2 = f(y_0 + k1 * dt/2, t_0 + dt/2)
                    for (int n = 0; n < NCOMP; ++n)
                        out(i, j, k, n) = in(i, j, k, n) + 0.5 * dt * k1[n];
                    const auto &k2 = geometric_source(r, alpha, rot_axis,
                                                      *parm, out, i, j);

                    //! k3 = f(y_0 + k2 * dt/2, t_0 + dt/2)
                    for (int n = 0; n < NCOMP; ++n)
                        out(i, j, k, n) = in(i, j, k, n) + 0.5 * dt * k2[n];
                    const auto &k3 = geometric_source(r, alpha, rot_axis,
                                                      *parm, out, i, j);

                    //! k4 = f(y_0 + k3 * dt, t_0 + dt)
                    for (int n = 0; n < NCOMP; ++n)
                        out(i, j, k, n) = in(i, j, k, n) + dt * k3[n];
                    const auto &k4 = geometric_source(r, alpha, rot_axis,
                                                      *parm, out, i, j);

                    //! y^new = y_0 + (k1 + 2k2 + 2k3 + k4) * dt/6
                    for (int n = 0; n < NCOMP; ++n)
                        out(i, j, k, n)
                            = in(i, j, k, n)
                              + (k1[n] + 2 * k2[n] + 2 * k3[n] + k4[n]) * dt
                                    / 6;
                });
        }
    }
#else
    amrex::ignore_unused(geom, dt, alpha, rot_axis, statein, stateout);
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