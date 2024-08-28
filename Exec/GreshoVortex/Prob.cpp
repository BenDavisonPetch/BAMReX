
#include <AMReX_BLassert.H>
#include <AMReX_Extension.H>
#include <AMReX_Math.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_PlotFileUtil.H>
#include <cmath>

#include "AmrLevelAdv.H"
#include "Euler/Euler.H"
#include "Euler/RiemannSolver.H"
#include "Prob.H"

using namespace amrex;

namespace details
{

Real u_theta(Real r)
{
    AMREX_ASSERT(r >= 0);
    return (0 <= r && r < 0.2) ? 5 * r : ((r < 0.4) ? 2 - 5 * r : 0);
}

Real p(Real r, Real p_0)
{
    return (0 <= r && r < 0.2)
               ? p_0 + (Real)25 / 2 * r * r
               : ((r < 0.4) ? p_0 + (Real)25 / 2 * r * r
                                  + 4 * (1 - 5 * r + log(5 * r))
                            : p_0 - 2 + 4 * log(2));
}

Real theta(Real x, Real y)
{
    return std::atan2(y, x);
}

Real r(Real x, Real y) { return sqrt(x * x + y * y); }

Real u_x(Real u_theta, Real theta) { return -u_theta * sin(theta); }

Real u_y(Real u_theta, Real theta) { return u_theta * cos(theta); }

}

/**
 * Initialize Data on Multifab
 *
 * \param S_tmp Pointer to Multifab where data is to be initialized.
 * \param geom Pointer to Multifab's geometry data.
 *
 */
void initdata(MultiFab &S_tmp, const Geometry &geom)
{
    Real M; // Mach number

    {
        ParmParse pp("init");
        pp.get("M", M);
    }

    const Real adia = AmrLevelAdv::h_parm->adiabatic;
    const Real eps  = AmrLevelAdv::h_parm->epsilon;

    AMREX_ALWAYS_ASSERT(eps == 1);

    const auto &prob_lo = geom.ProbLoArray();
    const auto &dx      = geom.CellSizeArray();

    // initial conditions
    const Real dens = 1;
    const Real p_0  = dens / (adia * M * M);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    {
        for (MFIter mfi(S_tmp, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const auto &bx    = mfi.growntilebox();
            const auto &state = S_tmp.array(mfi);
            ParallelFor(bx,
                        [=] AMREX_GPU_DEVICE(int i, int j, int k)
                        {
                            const Real x     = (i + 0.5) * dx[0] + prob_lo[0];
                            const Real y     = (j + 0.5) * dx[1] + prob_lo[1];
                            const Real r     = details::r(x, y);
                            const Real theta = details::theta(x, y);
                            const Real u_theta = details::u_theta(r);
                            const Real u_x     = details::u_x(u_theta, theta);
                            const Real u_y     = details::u_y(u_theta, theta);
                            const Real p       = details::p(r, p_0);
                            const GpuArray<Real, EULER_NCOMP> primv(
                                { dens, AMREX_D_DECL(u_x, u_y, 0), p });
                            const auto &consv
                                = consv_from_primv(primv, adia, eps);
                            AMREX_PRAGMA_SIMD
                            for (int n = 0; n < EULER_NCOMP; ++n)
                            {
                                state(i, j, k, n) = consv[n];
                            }
                        });
        }
    }
}