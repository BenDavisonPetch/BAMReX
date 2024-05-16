
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

AMREX_FORCE_INLINE AMREX_GPU_DEVICE Real density_integral(Real x)
{
    // indefinite integral of density wrt x
    return 19 * x / 8 - Math::sinpi(2 * x) / (4 * Math::pi<Real>())
           + Math::sinpi(4 * x) / (32 * Math::pi<Real>());
}

AMREX_FORCE_INLINE AMREX_GPU_DEVICE Real density_cell_avg(int i, Real dx,
                                                          Real prob_lo,
                                                          Real time)
{
    Real x_L = i * dx + prob_lo - time;
    Real x_R = (i + 1) * dx + prob_lo - time;
    return (density_integral(x_R) - density_integral(x_L)) / dx;
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

    const GpuArray<Real, AMREX_SPACEDIM> dx      = geom.CellSizeArray();
    const GpuArray<Real, AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();

    Real stop_time; // used when calculating exact solution

    {
        ParmParse pp2;
        pp2.get("stop_time", stop_time);
    }
    Real p = 1;
    {
        ParmParse pp("prob");
        pp.query("pressure", p);
    }

    const Real adiabatic = AmrLevelAdv::h_prob_parm->adiabatic;
    const Real epsilon   = AmrLevelAdv::h_prob_parm->epsilon;

    write_exact_solution(stop_time, p, adiabatic, epsilon);

    for (MFIter mfi(S_tmp); mfi.isValid(); ++mfi)
    {
        const Box          &box = mfi.tilebox();
        const Array4<Real> &phi = S_tmp.array(mfi);

        // Populate MultiFab data
        ParallelFor(box,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k)
                    {
                        const auto &consv = consv_from_primv(
                            { density_cell_avg(i, dx[0], prob_lo[0], 0),
                              AMREX_D_DECL(1, 0, 0), p },
                            adiabatic, epsilon);
                        for (int n = 0; n < EULER_NCOMP; ++n)
                            phi(i, j, k, n) = consv[n];
                    });
    }
}

void write_exact_solution(amrex::Real time, amrex::Real p, amrex::Real adiabatic,
                          amrex::Real epsilon)
{
    // We have an exact solution so we can just write it to file
    const int                      resolution = 1000;
    IntVect                        dom_lo(AMREX_D_DECL(0, 0, 0));
    IntVect                        dom_hi(AMREX_D_DECL(resolution - 1, 0, 0));
    Box                            domain(dom_lo, dom_hi);
    BoxArray                       ba(domain);
    DistributionMapping            dm(ba);
    MultiFab                       mf(ba, dm, AmrLevelAdv::NUM_STATE, 0);
    GpuArray<Real, AMREX_SPACEDIM> prob_lo({ AMREX_D_DECL(-1., -1., -1.) });
    RealBox    real_box(AMREX_D_DECL(-1., -1., -1.), AMREX_D_DECL(1., 1., 1.));
    Geometry   geom(domain, &real_box);
    const auto dx = geom.CellSizeArray();

    for (MFIter mfi(mf, true); mfi.isValid(); ++mfi)
    {
        const auto &bx  = mfi.tilebox();
        const auto &arr = mf.array(mfi);

        ParallelFor(
            bx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                const Real x     = (i + 0.5) * dx[0] + prob_lo[0];
                const auto consv = consv_from_primv(
                    { 2 + amrex::Math::powi<4>(amrex::Math::sinpi(x - time)),
                      AMREX_D_DECL(1, 0, 0), p },
                    adiabatic, epsilon);
                for (int n = 0; n < EULER_NCOMP; ++n)
                    arr(i, j, k, n) = consv[n];
            });
    }
    ParmParse   pp("amr");
    std::string plotfile_name;
    pp.get("plot_file", plotfile_name);
    WriteSingleLevelPlotfile(
        plotfile_name + "EXACT_SOLN", mf,
        { "density", AMREX_D_DECL("mom_x", "mom_y", "mom_z"), "energy" }, geom,
        time, 0);
}