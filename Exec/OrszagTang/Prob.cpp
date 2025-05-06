#include "Prob.H"

#include "AmrLevelAdv.H"
#include "System/Plasma/Plasma_K.H"
#include <AMReX_MFIter.H>
#include <AMReX_Math.H>
#include <AMReX_MultiFab.H>

/**
 * Initialize Data on Multifab
 *
 * \param S_tmp Pointer to Multifab where data is to be initialized.
 * \param geom Pointer to Multifab's geometry data.
 *
 */
using namespace amrex;

AMREX_GPU_DEVICE GpuArray<Real, PQInd::size> init_orzagtang_Q(Real x, Real y,
                                                              const Parm &parm)
{
    GpuArray<Real, PQInd::size> Q;
    Q[PQInd::RHO] = parm.adiabatic * parm.adiabatic;
    Q[PQInd::P]   = parm.adiabatic;
    Q[PQInd::U]   = -Math::sinpi(2 * y);
    Q[PQInd::V]   = Math::sinpi(2 * x);
    Q[PQInd::W]   = 0;
    Q[PQInd::BX]  = -Math::sinpi(2 * y);
    Q[PQInd::BY]  = Math::sinpi(4 * x);
    Q[PQInd::BZ]  = 0;
    mhd_fill_p_tot(Q);
    return Q;
}

void initdata(MultiFab &S_tmp, const Geometry &geom)
{
    const System *system = AmrLevelAdv::system;
    const Parm   *d_parm = AmrLevelAdv::d_parm;
    if (system->GetSystemType() != SystemType::plasma)
        throw RuntimeError("Orzag Tang vortex requires prob.system=plasma");

#ifdef AMREX_USE_OMP
#pragma parallel if (Gpu::notInLaunchRegion())
#endif
    {
        const auto &prob_lo = geom.ProbLoArray();
        const auto &dx      = geom.CellSizeArray();
        for (MFIter mfi(S_tmp, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box  &tbx = mfi.growntilebox();
            const auto &in  = S_tmp.array(mfi);
            ParallelFor(tbx,
                        [=] AMREX_GPU_DEVICE(int i, int j, int k)
                        {
                            const Real  x = (i + 0.5) * dx[0] + prob_lo[0];
                            const Real  y = (j + 0.5) * dx[1] + prob_lo[1];
                            const auto &Q = init_orzagtang_Q(x, y, *d_parm);
                            const auto &U = mhd_ptoc(Q, *d_parm);
                            for (int n = 0; n < PUInd::size; ++n)
                            {
                                in(i, j, k, n) = U[n];
                            }
                        });
        }
    }
}