
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
    Real lambda; // vortex strength parameter
    Real density_bg, u_bg, v_bg, w_bg, pressure_bg;
    Real x_c, y_c;
    Real stop_time; // used when calculating exact solution

    {
        ParmParse pp("init");
        pp.get("lambda", lambda);
        pp.get("density_bg", density_bg);
        pp.get("u_bg", u_bg);
        pp.get("v_bg", v_bg);
        pp.get("w_bg", w_bg);
        pp.get("pressure_bg", pressure_bg);
        pp.get("x_c", x_c);
        pp.get("y_c", y_c);

        ParmParse pp2;
        pp2.get("stop_time", stop_time);
    }

    const Real adiabatic = AmrLevelAdv::h_parm->adiabatic;
    const Real epsilon   = AmrLevelAdv::h_parm->epsilon;

    // initial conditions
    init_exact_solution(0, S_tmp, geom, density_bg, u_bg, v_bg, w_bg,
                        pressure_bg, x_c, y_c, lambda, adiabatic, epsilon);

    // Write exact solution
    MultiFab S_exact(S_tmp.boxArray(), S_tmp.DistributionMap(), EULER_NCOMP,
                     0);
    init_exact_solution(stop_time, S_exact, geom, density_bg, u_bg, v_bg, w_bg,
                        pressure_bg, x_c, y_c, lambda, adiabatic, epsilon);
    write_exact_solution(stop_time, S_exact, geom);
}

namespace details
{
AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
amrex::Real domain_wrap(amrex::Real coord, amrex::Real lo, amrex::Real hi);
}

void init_exact_solution(const amrex::Real time, amrex::MultiFab &S,
                         const amrex::Geometry &geom,
                         const amrex::Real density_bg, const amrex::Real u_bg,
                         const amrex::Real v_bg, const amrex::Real w_bg,
                         const amrex::Real pressure_bg, const amrex::Real x_c,
                         const amrex::Real y_c, const amrex::Real lambda,
                         const amrex::Real adiabatic,
                         const amrex::Real epsilon)
{
#if AMREX_SPACEDIM < 3
    amrex::ignore_unused(w_bg);
#endif

    const auto dx      = geom.CellSizeArray();
    const auto prob_lo = geom.ProbLoArray();
    const auto prob_hi = geom.ProbHiArray();

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    {
        const Real dT_prefactor = -(adiabatic - 1) * lambda * lambda
                                  / (8 * adiabatic * amrex::Math::pi<Real>()
                                     * amrex::Math::pi<Real>());
        const Real vel_prefactor = lambda / (2 * amrex::Math::pi<Real>());
        for (MFIter mfi(S, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box          &box = mfi.tilebox();
            const Array4<Real> &phi = S.array(mfi);

            // Populate MultiFab data
            ParallelFor(
                box,
                [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    const Real x_wrapped = details::domain_wrap(
                        prob_lo[0] + (i + Real(0.5)) * dx[0] - u_bg * time,
                        prob_lo[0], prob_hi[0]);
                    const Real y_wrapped = details::domain_wrap(
                        prob_lo[1] + (j + Real(0.5)) * dx[1] - v_bg * time,
                        prob_lo[1], prob_hi[1]);
                    const Real x    = x_wrapped - x_c;
                    const Real y    = y_wrapped - y_c;
                    const Real r_sq = x * x + y * y;
                    const Real dT   = dT_prefactor * exp(1 - r_sq);
                    const Real vel_mult
                        = vel_prefactor * exp((1 - r_sq) * 0.5);
                    const Real density
                        = density_bg + pow(1 + dT, 1 / (adiabatic - 1)) - 1;
                    const Real pressure
                        = pressure_bg
                          + pow(1 + dT, adiabatic / (adiabatic - 1)) - 1;
                    const Real  u     = u_bg - vel_mult * y;
                    const Real  v     = v_bg + vel_mult * x;
                    const auto &consv = consv_from_primv(
                        { density, AMREX_D_DECL(u, v, w_bg), pressure },
                        adiabatic, epsilon);
                    for (int n = 0; n < EULER_NCOMP; ++n)
                        phi(i, j, k, n) = consv[n];
                });
        }
    }
}

void write_exact_solution(const amrex::Real time, const amrex::MultiFab &soln,
                          const amrex::Geometry &geom)
{
    ParmParse   pp("amr");
    std::string plotfile_name;
    pp.get("plot_file", plotfile_name);
    WriteSingleLevelPlotfile(
        plotfile_name + "EXACT_SOLN", soln,
        { "density", AMREX_D_DECL("mom_x", "mom_y", "mom_z"), "energy" }, geom,
        time, 0);
}

namespace details
{
AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
amrex::Real domain_wrap(amrex::Real coord, amrex::Real lo, amrex::Real hi)
{
    Real size = hi - lo;
    if (coord < lo)
        return coord + ceil((lo - coord) / size) * size;
    else if (coord > hi)
        return coord - ceil((coord - hi) / size) * size;
    return coord;
}
}