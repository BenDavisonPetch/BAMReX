
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
        air_specific_gas_const, scale_factor, inner_radius, outer_radius,
        length;
    Array<Real, 3> vjet_c;
    ParmParse      ppinit("init");
    ppinit.get("exit_v_ratio", exit_v_ratio);
    ppinit.get("primary_exit_M", primary_exit_M);
    ppinit.get("ambient_p", ambient_p);
    ppinit.get("ambient_T", ambient_T);
    ppinit.get("air_specific_gas_const", air_specific_gas_const);

    ParmParse ppls("ls");
    ppls.get("scale_factor", scale_factor);
    ppls.get("inner_radius", inner_radius);
    ppls.get("outer_radius", outer_radius);
    ppls.get("length", length);
    ppls.get("nozzle_center", vjet_c);
    inner_radius *= scale_factor;
    outer_radius *= scale_factor;
    length *= scale_factor;
    // XDim3 jet_center({ vjet_c[0], vjet_c[1], vjet_c[2] });

    const Real adia = AmrLevelAdv::h_prob_parm->adiabatic;
    const Real eps  = AmrLevelAdv::h_prob_parm->epsilon;

    const Real dens = ambient_p / (air_specific_gas_const * ambient_T);
    // const Real c    = sqrt(ambient_p * adia / dens);
    // const Real v_p  = primary_exit_M * c;
    // const Real v_s  = exit_v_ratio * v_p;
    const GpuArray<Real, EULER_NCOMP> init_primv{ dens, AMREX_D_DECL(0, 0, 0),
                                                  ambient_p };
    // const GpuArray<Real, EULER_NCOMP> p_primv{ dens, AMREX_D_DECL(v_p, 0,
    // 0),
    //                                            ambient_p };
    // const GpuArray<Real, EULER_NCOMP> s_primv{ dens, AMREX_D_DECL(v_s, 0,
    // 0),
    //                                            ambient_p };

    const auto &init_consv = consv_from_primv(init_primv, adia, eps);
    // const auto &p_consv    = consv_from_primv(p_primv, adia, eps);
    // const auto &s_consv    = consv_from_primv(s_primv, adia, eps);

    // const auto &dx      = geom.CellSizeArray();
    // const auto &prob_lo = geom.ProbLoArray();

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
                        {
                            //                             const Real x = (i +
                            //                             0.5) * dx[0] +
                            //                             prob_lo[0]
                            //                                            -
                            //                                            jet_center.x;
                            //                             const Real y = (j +
                            //                             0.5) * dx[1] +
                            //                             prob_lo[1]
                            //                                            -
                            //                                            jet_center.y;
                            // #if AMREX_SPACEDIM == 2
                            //                             const Real r =
                            //                             fabs(y -
                            //                             jet_center.y);
                            // #elif AMREX_SPACEDIM == 3
                            //                             const Real z = (k +
                            //                             0.5) * dx[2] +
                            //                             prob_lo[2] -
                            //                             jet_center.z; const
                            //                             Real r = sqrt(y*y +
                            //                             z*z);
                            // #else
                            // #error "Spacedim should be 2 or 3!"
                            // #endif
                            //                             if (x < length && r
                            //                             < inner_radius)
                            //                                 arr(i, j, k, n)
                            //                                 = p_consv[n];
                            //                             else if (x < length
                            //                             && r < outer_radius)
                            //                                 arr(i, j, k, n)
                            //                                 = s_consv[n];
                            //                             else
                            arr(i, j, k, n) = init_consv[n];
                        });
        }
    }
}