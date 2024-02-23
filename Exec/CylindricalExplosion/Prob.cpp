
#include <AMReX_Math.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include <cmath>

#include "AmrLevelAdv.H"
#include "Euler/Euler.H"

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

    const GpuArray<Real, AMREX_SPACEDIM> dx      = geom.CellSizeArray();
    const GpuArray<Real, AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();

    Real density_in, pressure_in, density_out, pressure_out, radius;

    ParmParse pp("init");
    pp.get("density_in", density_in);
    pp.get("pressure_in", pressure_in);
    pp.get("density_out", density_out);
    pp.get("pressure_out", pressure_out);
    pp.query("radius", radius);

    const double adiabatic = AmrLevelAdv::h_prob_parm->adiabatic;

    GpuArray<Real, EULER_NCOMP> primv_in{
        density_in, AMREX_D_DECL(0, 0, 0), pressure_in
    };
    GpuArray<Real, EULER_NCOMP> primv_out{
        density_out, AMREX_D_DECL(0, 0, 0), pressure_out
    };

    const auto consv_L = consv_from_primv(primv_in, adiabatic);
    const auto consv_R = consv_from_primv(primv_out, adiabatic);

    for (MFIter mfi(S_tmp); mfi.isValid(); ++mfi)
    {
        const Box          &box = mfi.validbox();
        const Array4<Real> &phi = S_tmp.array(mfi);

        // Populate MultiFab data
        ParallelFor(box, AmrLevelAdv::NUM_STATE,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                    {
#if AMREX_SPACEDIM == 1
                        Real x = prob_lo[0] + (i + Real(0.5)) * dx[0];
                        Real r = x-1;
#elif AMREX_SPACEDIM == 2
                        Real x = prob_lo[0] + (i + Real(0.5)) * dx[0];
                        Real y = prob_lo[1] + (j + Real(0.5)) * dx[1];
                        Real r = sqrt((x-1)*(x-1) + (y-1)*(y-1));
#else
                        Real x = prob_lo[0] + (i + Real(0.5)) * dx[0];
                        Real y = prob_lo[1] + (j + Real(0.5)) * dx[1];
                        Real z = prob_lo[2] + (k + Real(0.5)) * dx[2];
                        Real r = sqrt((x-1)*(x-1) + (y-1)*(y-1) + (z-1)*(z-1));
#endif
                        if (r < radius)
                            phi(i, j, k, n) = consv_L[n];
                        else
                            phi(i, j, k, n) = consv_R[n];
                    });
    }
}
