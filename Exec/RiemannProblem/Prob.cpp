
#include <AMReX_Math.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_PlotFileUtil.H>
#include <cmath>

#include "AmrLevelAdv.H"
#include "Prob.H"
#include "System/Euler/Euler.H"
#include "System/Euler/RiemannSolver.H"
#include "System/Plasma/Plasma_K.H"

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
    switch (AmrLevelAdv::system->GetSystemType())
    {
    case SystemType::fluid: initdata_euler(S_tmp, geom); break;
    case SystemType::plasma: initdata_mhd(S_tmp, geom); break;
    }
}

void initdata_euler(MultiFab &S_tmp, const Geometry &geom)
{

    const GpuArray<Real, AMREX_SPACEDIM> dx      = geom.CellSizeArray();
    const GpuArray<Real, AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();
    const GpuArray<Real, AMREX_SPACEDIM> prob_hi = geom.ProbHiArray();

    Real density_L, velocity_L, pressure_L, density_R, velocity_R, pressure_R;
    Real rotation = 0;
    Real stop_time; // used when calculating exact solution

    {
        ParmParse pp("init");
        pp.get("density_L", density_L);
        pp.get("velocity_L", velocity_L);
        pp.get("pressure_L", pressure_L);
        pp.get("density_R", density_R);
        pp.get("velocity_R", velocity_R);
        pp.get("pressure_R", pressure_R);
        pp.query("rotation", rotation);
        rotation *= Math::pi<Real>() / Real(180);

        ParmParse pp2;
        pp2.get("stop_time", stop_time);
    }

    const Real adiabatic = AmrLevelAdv::h_parm->adiabatic;
    const Real epsilon   = AmrLevelAdv::h_parm->epsilon;

#if AMREX_SPACEDIM == 1
    GpuArray<Real, EULER_NCOMP> primv_L{ density_L,
                                         AMREX_D_DECL(velocity_L, 0, 0),
                                         pressure_L };
    GpuArray<Real, EULER_NCOMP> primv_R{ density_R,
                                         AMREX_D_DECL(velocity_R, 0, 0),
                                         pressure_R };

    write_exact_solution(stop_time, primv_L, primv_R, adiabatic, adiabatic,
                         epsilon);
#else
    GpuArray<Real, EULER_NCOMP> primv_L{ density_L,
                                         AMREX_D_DECL(0, velocity_L, 0),
                                         pressure_L };
    GpuArray<Real, EULER_NCOMP> primv_R{ density_R,
                                         AMREX_D_DECL(0, velocity_R, 0),
                                         pressure_R };

    GpuArray<Real, EULER_NCOMP> primv_L_x_oriented{
        density_L, AMREX_D_DECL(velocity_L, 0, 0), pressure_L
    };
    GpuArray<Real, EULER_NCOMP> primv_R_x_oriented{
        density_R, AMREX_D_DECL(velocity_R, 0, 0), pressure_R
    };

    write_exact_solution(stop_time, primv_L_x_oriented, primv_R_x_oriented,
                         adiabatic, adiabatic);
#endif

    const auto consv_L = consv_from_primv(primv_L, adiabatic, epsilon);
    const auto consv_R = consv_from_primv(primv_R, adiabatic, epsilon);

    for (MFIter mfi(S_tmp); mfi.isValid(); ++mfi)
    {
        const Box          &box = mfi.validbox();
        const Array4<Real> &phi = S_tmp.array(mfi);

        // Populate MultiFab data
        ParallelFor(box, EULER_NCOMP,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                    {
#if AMREX_SPACEDIM == 1
                        const Real x_mid = (prob_lo[0] + prob_hi[0]) * 0.5;
                        Real       x = prob_lo[0] + (i + Real(0.5)) * dx[0];
                        if (x < x_mid)
                            phi(i, j, k, n) = consv_L[n];
                        else
                            phi(i, j, k, n) = consv_R[n];
#else
                        const Real y_mid = (prob_lo[1] + prob_hi[1]) * 0.5;
                        Real x = prob_lo[0] + (i + Real(0.5)) * dx[0];
                        Real y = prob_lo[1] + (j + Real(0.5)) * dx[1];
                        if (y < 0.5)
                            phi(i, j, k, n) = consv_L[n];
                        else
                            phi(i, j, k, n) = consv_R[n];
#endif
                    });
    }
}

void initdata_mhd(MultiFab &S_tmp, const Geometry &geom)
{
    const GpuArray<Real, AMREX_SPACEDIM> dx      = geom.CellSizeArray();
    const GpuArray<Real, AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();
    const GpuArray<Real, AMREX_SPACEDIM> prob_hi = geom.ProbHiArray();

    Real           density_L, pressure_L, density_R, pressure_R;
    Array<Real, 3> velocity_L, velocity_R, B_L, B_R;
    Real           rotation = 0;
    Real           stop_time; // used when calculating exact solution

    {
        ParmParse pp("init");
        pp.get("density_L", density_L);
        pp.get("velocity_L", velocity_L);
        pp.get("pressure_L", pressure_L);
        pp.get("B_L", B_L);
        pp.get("density_R", density_R);
        pp.get("velocity_R", velocity_R);
        pp.get("pressure_R", pressure_R);
        pp.get("B_R", B_R);
        pp.query("rotation", rotation);
        rotation *= Math::pi<Real>() / Real(180);

        ParmParse pp2;
        pp2.get("stop_time", stop_time);
    }
    const Real p_t_L
        = pressure_L
          + 0.5 * (B_L[0] * B_L[0] + B_L[1] * B_L[1] + B_L[2] * B_L[2]);
    const Real p_t_R
        = pressure_R
          + 0.5 * (B_R[0] * B_R[0] + B_R[1] * B_R[1] + B_R[2] * B_R[2]);

    GpuArray<Real, PQInd::size> Q_L{
        density_L, velocity_L[0], velocity_L[1], velocity_L[2], pressure_L,
        p_t_L,     B_L[0],        B_L[1],        B_L[2]
    };
    GpuArray<Real, PQInd::size> Q_R{
        density_R, velocity_R[0], velocity_R[1], velocity_R[2], pressure_R,
        p_t_R,     B_R[0],        B_R[1],        B_R[2]
    };

    const auto consv_L = mhd_ptoc(Q_L, *AmrLevelAdv::h_parm);
    const auto consv_R = mhd_ptoc(Q_R, *AmrLevelAdv::h_parm);

    for (MFIter mfi(S_tmp); mfi.isValid(); ++mfi)
    {
        const Box          &box = mfi.validbox();
        const Array4<Real> &phi = S_tmp.array(mfi);

        // Populate MultiFab data
        ParallelFor(box, PUInd::size,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                    {
#if AMREX_SPACEDIM == 1
                        const Real x_mid = (prob_lo[0] + prob_hi[0]) * 0.5;
                        Real       x = prob_lo[0] + (i + Real(0.5)) * dx[0];
                        if (x < x_mid)
                            phi(i, j, k, n) = consv_L[n];
                        else
                            phi(i, j, k, n) = consv_R[n];
#else
                        const Real y_mid = (prob_lo[1] + prob_hi[1]) * 0.5;
                        Real x = prob_lo[0] + (i + Real(0.5)) * dx[0];
                        Real y = prob_lo[1] + (j + Real(0.5)) * dx[1];
                        if (y < y_mid)
                            phi(i, j, k, n) = consv_L[n];
                        else
                            phi(i, j, k, n) = consv_R[n];
#endif
                    });
    }
}

void write_exact_solution(
    const amrex::Real                                time,
    const amrex::GpuArray<amrex::Real, EULER_NCOMP> &primv_L,
    const amrex::GpuArray<amrex::Real, EULER_NCOMP> &primv_R,
    const amrex::Real adiabatic_L, const amrex::Real adiabatic_R,
    const amrex::Real epsilon)
{
    // We basically just do a 1D simulation here and write it to file
    const int                      resolution = 1000;
    IntVect                        dom_lo(AMREX_D_DECL(0, 0, 0));
    IntVect                        dom_hi(AMREX_D_DECL(resolution - 1, 0, 0));
    Box                            domain(dom_lo, dom_hi);
    BoxArray                       ba(domain);
    DistributionMapping            dm(ba);
    MultiFab                       mf(ba, dm, EULER_NCOMP, 0);
    GpuArray<Real, AMREX_SPACEDIM> prob_lo({ AMREX_D_DECL(0., 0., 0.) });
    RealBox    real_box(AMREX_D_DECL(0., 0., 0.), AMREX_D_DECL(1., 1., 1.));
    Geometry   geom(domain, &real_box);
    const auto dx = geom.CellSizeArray();

    for (MFIter mfi(mf, true); mfi.isValid(); ++mfi)
    {
        const auto &bx  = mfi.tilebox();
        const auto &arr = mf.array(mfi);

        EulerExactRiemannSolver<0> rp_solver(primv_L, primv_R, adiabatic_L,
                                             adiabatic_R, epsilon);
        rp_solver.fill(time, 0.5, bx, dx, prob_lo, arr);
    }
    ParmParse   pp("amr");
    std::string plotfile_name;
    pp.get("plot_file", plotfile_name);
    WriteSingleLevelPlotfile(
        plotfile_name + "EXACT_SOLN", mf,
        { "density", AMREX_D_DECL("mom_x", "mom_y", "mom_z"), "energy" }, geom,
        time, 0);
}