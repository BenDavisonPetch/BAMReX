
#include <AMReX_Math.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include <cmath>

#include "AmrLevelAdv.H"
#include "System/Euler/Euler.H"

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
    AMREX_ASSERT(AMREX_SPACEDIM > 1);
    const GpuArray<Real, AMREX_SPACEDIM> dx      = geom.CellSizeArray();
    const GpuArray<Real, AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();

    Real density_TR, vel_x_TR, vel_y_TR, pressure_TR;
    Real density_TL, vel_x_TL, vel_y_TL, pressure_TL;
    Real density_BL, vel_x_BL, vel_y_BL, pressure_BL;
    Real density_BR, vel_x_BR, vel_y_BR, pressure_BR;

    ParmParse pp("init");
    pp.get("density_TR", density_TR);
    pp.get("velocity_x_TR", vel_x_TR);
    pp.get("velocity_y_TR", vel_y_TR);
    pp.get("pressure_TR", pressure_TR);

    pp.get("density_TL", density_TL);
    pp.get("velocity_x_TL", vel_x_TL);
    pp.get("velocity_y_TL", vel_y_TL);
    pp.get("pressure_TL", pressure_TL);

    pp.get("density_BL", density_BL);
    pp.get("velocity_x_BL", vel_x_BL);
    pp.get("velocity_y_BL", vel_y_BL);
    pp.get("pressure_BL", pressure_BL);

    pp.get("density_BR", density_BR);
    pp.get("velocity_x_BR", vel_x_BR);
    pp.get("velocity_y_BR", vel_y_BR);
    pp.get("pressure_BR", pressure_BR);

    const Real adiabatic = AmrLevelAdv::h_parm->adiabatic;
    const Real epsilon   = AmrLevelAdv::h_parm->epsilon;

    GpuArray<Real, EULER_NCOMP> primv_TR{ density_TR,
                                          AMREX_D_DECL(vel_x_TR, vel_y_TR, 0),
                                          pressure_TR };

    GpuArray<Real, EULER_NCOMP> primv_TL{ density_TL,
                                          AMREX_D_DECL(vel_x_TL, vel_y_TL, 0),
                                          pressure_TL };

    GpuArray<Real, EULER_NCOMP> primv_BL{ density_BL,
                                          AMREX_D_DECL(vel_x_BL, vel_y_BL, 0),
                                          pressure_BL };

    GpuArray<Real, EULER_NCOMP> primv_BR{ density_BR,
                                          AMREX_D_DECL(vel_x_BR, vel_y_BR, 0),
                                          pressure_BR };

    const auto consv_TR = consv_from_primv(primv_TR, adiabatic, epsilon);
    const auto consv_TL = consv_from_primv(primv_TL, adiabatic, epsilon);
    const auto consv_BL = consv_from_primv(primv_BL, adiabatic, epsilon);
    const auto consv_BR = consv_from_primv(primv_BR, adiabatic, epsilon);

    for (MFIter mfi(S_tmp); mfi.isValid(); ++mfi)
    {
        const Box          &box = mfi.validbox();
        const Array4<Real> &phi = S_tmp.array(mfi);

        // Populate MultiFab data
        ParallelFor(box, EULER_NCOMP,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                    {
                        Real x = prob_lo[0] + (i + Real(0.5)) * dx[0];
                        Real y = prob_lo[1] + (j + Real(0.5)) * dx[1];
                        if (x > 0.5 && y > 0.5)
                            phi(i, j, k, n) = consv_TR[n];
                        else if (x < 0.5 && y > 0.5)
                            phi(i, j, k, n) = consv_TL[n];
                        else if (x < 0.5 && y < 0.5)
                            phi(i, j, k, n) = consv_BL[n];
                        else
                            phi(i, j, k, n) = consv_BR[n];
                    });
    }
}
