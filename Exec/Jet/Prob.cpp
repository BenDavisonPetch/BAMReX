
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

void initdata(MultiFab &S_tmp, [[maybe_unused]] const Geometry &geom)
{
    const Real adia = AmrLevelAdv::h_parm->adiabatic;
    const Real eps  = AmrLevelAdv::h_parm->epsilon;

    Real exit_v_ratio, primary_exit_M, p_amb, T_amb, T_p, T_s, R_spec,
        inner_radius, outer_radius, scale_factor;
    ParmParse ppinit("init");
    ppinit.get("exit_v_ratio", exit_v_ratio);
    ppinit.get("primary_exit_M", primary_exit_M);
    ppinit.get("p_ambient", p_amb);
    ppinit.get("T_ambient", T_amb);
    ppinit.get("T_p", T_p);
    ppinit.get("T_s", T_s);

    ppinit.get("R_spec", R_spec);

    // RealArray center;
    ParmParse ppls("ls");
    ppls.get("inner_radius", inner_radius);
    ppls.get("outer_radius", outer_radius);
    // ppls.get("nozzle_center", center);
    ppls.get("scale_factor", scale_factor);
    inner_radius *= scale_factor;
    outer_radius *= scale_factor;
    // center doesn't get scaled

    const Real dens_amb = p_amb / (R_spec * T_amb);
    // const Real c        = sqrt(p_amb * adia / dens_amb);
    // const Real v_p      = primary_exit_M * c;
    // const Real v_s      = exit_v_ratio * v_p;

    const GpuArray<Real, EULER_NCOMP> init_primv{ dens_amb,
                                                  AMREX_D_DECL(0, 0, 0),
                                                  p_amb };

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