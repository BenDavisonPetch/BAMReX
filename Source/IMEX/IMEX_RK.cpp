#include "IMEX_RK.H"
#include "AmrLevelAdv.H"
#include "Euler/Euler.H"
#include "Euler/NComp.H"
#include "Fluxes/Update.H"
#include "IMEX/ButcherTableau.H"
#include "IMEX/IMEXSettings.H"
#include "IMEX_K/euler_K.H"

#include <AMReX_AmrLevel.H>
#include <AMReX_Arena.H>
#include <AMReX_BCUtil.H>
#include <AMReX_Extension.H>
#include <AMReX_GpuControl.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_REAL.H>

using namespace amrex;

AMREX_GPU_HOST
void advance_imex_rk(const amrex::Real time, const amrex::Geometry &geom,
                     const amrex::MultiFab &statein, amrex::MultiFab &stateout,
                     amrex::MultiFab                               &pressure,
                     amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> &fluxes,
                     const amrex::Real dt, const bamrexBCData &bc_data,
                     const IMEXSettings settings)
{
    BL_PROFILE_REGION("advance_imex_rk()");
    BL_PROFILE("advance_imex_rk()");
    AMREX_ASSERT(!pressure.contains_nan()); // pressure must be initialised!
    AMREX_ASSERT(pressure.nGrow() >= 1);

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
        if (settings.verbose)
            Print() << "Starting RK stage " << stage << "..." << std::endl;
        // Put together Qe and Qs
        const auto dt_imp = tableau.get_A_imp_row(stage, dt);
        const auto dt_exp = tableau.get_A_exp_row(stage, dt);
        conservative_update(geom, statein, stage_fluxes, statein_imp,
                            statein_exp[stage % 2], dt_imp, dt_exp, stage);

        bc_data.fill_consv_boundary(geom, statein_imp, time);
        bc_data.fill_consv_boundary(geom, statein_exp[stage % 2], time);

        const Real adt = tableau.get_A_imp()[stage][stage] * dt;

        // Retrieve pressure
        if (stage > 0)
        {
            compute_pressure(statein_exp[stage % 2],
                             statein_exp[(stage - 1) % 2], pressure, geom, adt,
                             settings);
            bc_data.fill_pressure_boundary(geom, pressure,
                                           statein_exp[stage % 2], 0);
        }

        // Compute stage flux for this step
        compute_flux_imex_o1(time, geom, statein_imp, statein_exp[stage % 2],
                             pressure, stage_fluxes[stage], adt, bc_data,
                             settings);

        if (settings.verbose)
            Print() << "Computed flux for RK stage " << stage << std::endl;
    }

    // Now we've computed all of the fluxes we sum to the get the final state
    sum_fluxes(stage_fluxes, fluxes, tableau.get_b_imp());
    // And update
    conservative_update(geom, statein, fluxes, stateout, dt);
}

AMREX_GPU_HOST
void advance_imex_rk_stab(
    const amrex::Real time, const amrex::Geometry &geom,
    const amrex::MultiFab &statein, amrex::MultiFab &stateout,
    amrex::MultiFab                               &pressure,
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> &fluxes,
    const amrex::Real dt, const bamrexBCData &bc_data,
    const IMEXSettings settings)
{
    BL_PROFILE("advance_imex_rk_stab()");
    if (!settings.stabilize)
    {
        advance_imex_rk(time, geom, statein, stateout, pressure, fluxes, dt,
                        bc_data, settings);
        return;
    }

    // We need a copy of the pressure for later
    MultiFab pressure_o1(pressure.boxArray(), pressure.DistributionMap(),
                         pressure.nComp(), pressure.nGrow(), MFInfo(),
                         pressure.Factory());
    pressure_o1.ParallelCopy(pressure, 0, 0, pressure.nComp(),
                             pressure.nGrow(), pressure.nGrow());

    advance_imex_rk(time, geom, statein, stateout, pressure, fluxes, dt,
                    bc_data, settings);

    const IMEXButcherTableau tableau(settings.butcher_tableau);
    if (tableau.order() <= 1)
        return;

    MultiFab shock_indicator;
    shock_indicator.define(statein.boxArray(), statein.DistributionMap(), 1,
                           0);

    Real indicator_max = compute_shock_indicator(
        geom, shock_indicator, stateout, bc_data, settings.k1);

    if (indicator_max <= 0)
        return;

    if (settings.verbose)
        Print() << "\tAt least one cell is troubled. Computing first order "
                   "solution..."
                << std::endl;

    // Compute 1st order solution
    MultiFab stateout_o1(stateout.boxArray(), stateout.DistributionMap(),
                         stateout.nComp(), stateout.nGrow());
    Array<MultiFab, AMREX_SPACEDIM> fluxes_o1;
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
        fluxes_o1[d].define(fluxes[d].boxArray(), fluxes[d].DistributionMap(),
                            fluxes[d].nComp(), fluxes[d].nGrow());
    IMEXSettings settings_o1(settings);
    settings_o1.advection_flux  = IMEXSettings::rusanov;
    settings_o1.butcher_tableau = IMEXButcherTableau::SP111;
    advance_imex_rk(time, geom, statein, stateout_o1, pressure_o1, fluxes_o1,
                    dt, bc_data, settings_o1);

    // Convex combo
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    {
        for (MFIter mfi(stateout, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            // IMPORTANT: ghost cells for pressure are not filled after the end
            // of this routine
            const auto &bx        = mfi.tilebox();
            const auto &indicator = shock_indicator.const_array(mfi);
            const auto &out
                = stateout.array(mfi); // currently contains high order soln
            const auto &o1   = stateout_o1.const_array(mfi);
            const auto &p    = pressure.array(mfi);
            const auto &p_o1 = pressure_o1.const_array(mfi);

            ParallelFor(bx,
                        [=] AMREX_GPU_DEVICE(int i, int j, int k)
                        {
                            AMREX_PRAGMA_SIMD
                            for (int n = 0; n < EULER_NCOMP; ++n)
                                out(i, j, k, n)
                                    = (indicator(i, j, k) > 0)
                                          ? indicator(i, j, k) * o1(i, j, k, n)
                                                + (1 - indicator(i, j, k))
                                                      * out(i, j, k, n)
                                          : out(i, j, k, n);
                            p(i, j, k)
                                = (indicator(i, j, k) > 0)
                                      ? indicator(i, j, k) * p_o1(i, j, k)
                                            + (1 - indicator(i, j, k))
                                                  * p(i, j, k)
                                      : p(i, j, k);
                        });
        }
    }
}

AMREX_GPU_HOST
void sum_fluxes(
    const amrex::Vector<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> >
                                                  &fluxes,
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> &dst,
    const amrex::Vector<amrex::Real>              &weighting)
{
    BL_PROFILE("sum_fluxes()");
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

AMREX_GPU_HOST
amrex::Real compute_shock_indicator(const amrex::Geometry &geom,
                                    amrex::MultiFab       &dst,
                                    const amrex::MultiFab &state,
                                    const bamrexBCData    &bc_data,
                                    const amrex::Real      k1)
{
    BL_PROFILE("compute_shock_indicator()");
#ifdef DEBUG
    ParmParse pp("amr");
    int       max_level;
    pp.get("max_level", max_level);
    AMREX_ASSERT(
        max_level
        == 0); // boundary filling does not fill course-fine boundaries
#endif

    MultiFab state_grow(state.boxArray(), state.DistributionMap(),
                        state.nComp(), 2);
    state_grow.ParallelCopy(state, 0, 0, state.nComp(), 0, 0);
    bc_data.fill_consv_boundary(geom, state_grow, 0);

    const Real eps  = AmrLevelAdv::h_prob_parm->epsilon;
    const Real adia = AmrLevelAdv::h_prob_parm->adiabatic;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    {
        const auto idx = geom.InvCellSizeArray();

        for (MFIter mfi(state_grow, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const auto &bx    = mfi.tilebox();
            const auto &bx_g1 = amrex::grow(bx, 1);

            // shock indicator before neumann neighbour max operation
            FArrayBox   indicator_premax(bx_g1, 1, The_Async_Arena());
            const auto &ind_premax = indicator_premax.array();
            const auto &ind        = dst.array(mfi);
            const auto &arr        = state_grow.const_array(mfi);

            ParallelFor(
                bx_g1,
                [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    const auto primv
                        = primv_from_consv(arr, adia, eps, i, j, k);
                    auto c_min = sound_speed(primv, adia, eps);
                    // Neumann neighbours
                    GpuArray<IntVect, AMREX_SPACEDIM * 2> nn;
                    AMREX_D_TERM(nn[0] = IntVect(AMREX_D_DECL(i + 1, j, k));
                                 nn[1] = IntVect(AMREX_D_DECL(i - 1, j, k));
                                 , nn[2] = IntVect(AMREX_D_DECL(i, j + 1, k));
                                 nn[3] = IntVect(AMREX_D_DECL(i, j - 1, k));
                                 , nn[4] = IntVect(AMREX_D_DECL(i, j, k + 1));
                                 nn[5] = IntVect(AMREX_D_DECL(i, j, k - 1));)
                    // Velocities of neumann neighbours
                    GpuArray<Real, AMREX_SPACEDIM * 2> u_nn;
                    for (int inn = 0; inn < AMREX_SPACEDIM * 2; ++inn)
                    {
                        const auto primv_n
                            = primv_from_consv(arr, adia, eps, nn[inn]);
                        c_min = min(c_min, sound_speed(primv_n, adia, eps));
                        u_nn[inn] = primv_n[(inn / 2) + 1];
                    }

                    Real div_u
                        = 0.5
                          * (AMREX_D_TERM(idx[0] * (u_nn[0] - u_nn[1]),
                                          +idx[1] * (u_nn[2] - u_nn[3]),
                                          +idx[2] * (u_nn[4] - u_nn[5])));

                    ind_premax(i, j, k) = fmin(
                        1, fmax(0, -(div_u + k1 * c_min) / (k1 * c_min)));
                });

            ParallelFor(bx,
                        [=] AMREX_GPU_DEVICE(int i, int j, int k)
                        {
                            auto          local_ind_max = ind_premax(i, j, k);
                            constexpr int j_g = (AMREX_SPACEDIM >= 2) ? 1 : 0;
                            constexpr int k_g = (AMREX_SPACEDIM >= 3) ? 1 : 0;
                            for (int ii = i - 1; ii <= i + 1; ++ii)
                            {
                                for (int jj = j - j_g; jj <= j + j_g; ++jj)
                                {
                                    for (int kk = k - k_g; kk <= k + k_g; ++kk)
                                    {
                                        local_ind_max
                                            = max(local_ind_max,
                                                  ind_premax(ii, jj, kk));
                                    }
                                }
                            }
                            ind(i, j, k) = local_ind_max;
                        });
        }
    }
    auto ind_max = dst.max(0);
    ParallelDescriptor::ReduceRealMax(ind_max);
    return ind_max;
}
