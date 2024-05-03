#include "IMEX_RK.H"
#include "AmrLevelAdv.H"
#include "Euler/NComp.H"
#include "Fluxes/Update.H"
#include "IMEX_K/euler_K.H"

#include <AMReX_AmrLevel.H>
#include <AMReX_BCUtil.H>
#include <AMReX_GpuControl.H>
#include <AMReX_MFIter.H>
#include <AMReX_REAL.H>

using namespace amrex;

AMREX_GPU_HOST
void advance_imex_rk(const amrex::Real time, const amrex::Geometry &geom,
                     const amrex::MultiFab &statein, amrex::MultiFab &stateout,
                     amrex::MultiFab                               &pressure,
                     amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> &fluxes,
                     const amrex::Real                              dt,
                     const amrex::Vector<amrex::BCRec>             &domainbcs,
                     const IMEXSettings                             settings)
{
    AMREX_ASSERT(!pressure.contains_nan()); // pressure must be initialised!
    AMREX_ASSERT(pressure.nGrow() >= 2);

    const IMEXButcherTableau tableau(settings.butcher_tableau);
    const int                s = tableau.n_steps_imp();

    Vector<Array<MultiFab, AMREX_SPACEDIM> > stage_fluxes(s);
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
        for (int is = 0; is < s; ++is)
        {
            stage_fluxes[is][d].define(fluxes[d].boxArray(),
                                       fluxes[d].DistributionMap(),
                                       fluxes[d].nComp(), fluxes[d].nGrow());
        }

    Print() << "Allocated stage fluxes" << std::endl;

    MultiFab                statein_imp; // tildeQi
    std::array<MultiFab, 2> statein_exp; // stores Qe at the previous stage as
                                         // well as the current one
    statein_imp.define(statein.boxArray(), statein.DistributionMap(),
                       statein.nComp(), statein.nGrow());
    statein_exp[0].define(statein.boxArray(), statein.DistributionMap(),
                          statein.nComp(), statein.nGrow());
    if (s > 1)
        statein_exp[1].define(statein.boxArray(), statein.DistributionMap(),
                              statein.nComp(), statein.nGrow());

    for (int stage = 0; stage < (int)s; ++stage)
    {
        if (settings.verbose) Print() << "Starting RK stage " << stage  << "..." << std::endl;
        // Put together Qe and Qs
        conservative_update(geom, statein, stage_fluxes, statein_imp,
                            tableau.get_A_imp_row(stage, dt), stage);
        conservative_update(geom, statein, stage_fluxes,
                            statein_exp[stage % 2],
                            tableau.get_A_exp_row(stage, dt), stage);
        statein_exp[stage % 2].FillBoundary_nowait(geom.periodicity());
        statein_imp.FillBoundary_nowait(geom.periodicity());
        statein_exp[stage % 2].FillBoundary_finish();
        FillDomainBoundary(statein_exp[stage % 2], geom, domainbcs);
        statein_imp.FillBoundary_finish();
        FillDomainBoundary(statein_imp, geom, domainbcs);

        // Retrieve pressure
        if (stage > 0)
        {
            compute_pressure(statein_exp[stage % 2],
                             statein_exp[(stage - 1) % 2], pressure, settings);
        }

        // Compute stage flux for this step
        compute_flux_imex_o1(time, geom, statein_imp, statein_exp[stage % 2],
                             pressure, stage_fluxes[stage],
                             tableau.get_A_imp()[stage][stage] * dt, domainbcs,
                             settings);

        if (settings.verbose) Print() << "Computed flux for RK stage " << stage << std::endl;
    }

    // Now we've computed all of the fluxes we sum to the get the final state
    sum_fluxes(stage_fluxes, fluxes, tableau.get_b_imp());
    // And update
    conservative_update(geom, statein, fluxes, stateout, dt);
}

AMREX_GPU_HOST
void sum_fluxes(
    const amrex::Vector<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> >
                                                  &fluxes,
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> &dst,
    const amrex::Vector<amrex::Real>              &weighting)
{
    AMREX_ASSERT(fluxes.size() == weighting.size());
    int size = fluxes.size();
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    {
        amrex::Vector<Array4<const Real> > fluxarr(size);

        for (int d = 0; d < AMREX_SPACEDIM; ++d)
            for (MFIter mfi(dst[d], TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                const auto &bx = mfi.growntilebox();

                const auto &dstarr = dst[d].array(mfi);
                // TODO: make this work on GPU
                for (int iN = 0; iN < size; ++iN)
                    fluxarr[iN] = fluxes[iN][d].const_array(mfi);

                ParallelFor(bx, dst[d].nComp(),
                            [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                            {
                                dstarr(i, j, k, n)
                                    = fluxarr[0](i, j, k, n) * weighting[0];
                                for (int iN = 1; iN < size; ++iN)
                                {
                                    dstarr(i, j, k, n)
                                        += fluxarr[iN](i, j, k, n)
                                           * weighting[iN];
                                }
                            });
            }
    }
}
