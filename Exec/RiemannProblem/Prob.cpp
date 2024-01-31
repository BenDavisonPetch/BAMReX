
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

    Real density_L, velocity_L, pressure_L, density_R, velocity_R, pressure_R;
    Real rotation = 0;

    ParmParse pp("init");
    pp.get("density_L", density_L);
    pp.get("velocity_L", velocity_L);
    pp.get("pressure_L", pressure_L);
    pp.get("density_R", density_R);
    pp.get("velocity_R", velocity_R);
    pp.get("pressure_R", pressure_R);
    pp.query("rotation", rotation);
    rotation *= Math::pi<Real>() / Real(180);

    const double adiabatic = AmrLevelAdv::h_prob_parm->adiabatic;

#if AMREX_SPACEDIM == 1
    GpuArray<const Real, MAX_STATE_SIZE> primv_L{
        density_L, AMREX_D_DECL(velocity_L, 0, 0), pressure_L
    };
    GpuArray<const Real, MAX_STATE_SIZE> primv_R{
        density_R, AMREX_D_DECL(velocity_R, 0, 0), pressure_R
    };
#else
    GpuArray<const Real, MAX_STATE_SIZE> primv_L{
        density_L, AMREX_D_DECL(0, velocity_L, 0), pressure_L
    };
    GpuArray<const Real, MAX_STATE_SIZE> primv_R{
        density_R, AMREX_D_DECL(0, velocity_R, 0), pressure_R
    };
#endif

    const auto consv_L = conservative_from_primitive(primv_L, adiabatic);
    const auto consv_R = conservative_from_primitive(primv_R, adiabatic);

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
                        if (x < 0.5)
                            phi(i, j, k, n) = consv_L[n];
                        else
                            phi(i, j, k, n) = consv_R[n];
#else
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
