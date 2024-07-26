
#include <AMReX_MFIter.H>
#include <AMReX_Math.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_PlotFileUtil.H>
#include <cmath>

#include "AmrLevelAdv.H"
#include "Euler/Euler.H"
#include "Euler/RiemannSolver.H"
#include "Prob.H"

/**
 * Initialize Data on Multifab
 *
 * \param S_tmp Pointer to Multifab where data is to be initialized.
 * \param geom Pointer to Multifab's geometry data.
 *
 */
using namespace amrex;

void initdata(MultiFab &S_tmp, const Geometry &geom)
{
    Real exit_v_ratio, primary_exit_M, ambient_p, ambient_T,
        air_specific_gas_const, inner_radius, outer_radius;
    ParmParse ppinit("init");
    ppinit.get("exit_v_ratio", exit_v_ratio);
    ppinit.get("primary_exit_M", primary_exit_M);
    ppinit.get("ambient_p", ambient_p);
    ppinit.get("ambient_T", ambient_T);
    ppinit.get("air_specific_gas_const", air_specific_gas_const);

    const Real adia = AmrLevelAdv::h_prob_parm->adiabatic;
    const Real eps  = AmrLevelAdv::h_prob_parm->epsilon;

    const Real dens = ambient_p / (air_specific_gas_const * ambient_T);
    // const Real c    = sqrt(ambient_p * adia / dens);
    // const Real v_p  = primary_exit_M * c;
    // const Real v_s  = exit_v_ratio;
    const GpuArray<Real, EULER_NCOMP> init_primv{ dens, AMREX_D_DECL(0, 0, 0),
                                                  ambient_p };
    const auto &init_consv = consv_from_primv(init_primv, adia, eps);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    {
        for (MFIter mfi(S_tmp, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const auto &bx  = mfi.tilebox();
            const auto &arr = S_tmp.array(mfi);

            ParallelFor(bx, EULER_NCOMP,
                        [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                        { arr(i, j, k, n) = init_consv[n]; });
        }
    }
}