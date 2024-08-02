#include "IMEX_O1.H"
#include "AmrLevelAdv.H"
#include "GFM/RigidBody.H"
#include "IMEX/ButcherTableau.H"
#include "IMEX/IMEXSettings.H"
#include "IMEX_Fluxes.H"
#include "IMEX_K/euler_K.H"
#include "IMEX_O1_RB.H"
#include "LinearSolver/MLALaplacianGrad.H"
#include "MFIterLoop.H"

#include <AMReX_Arena.H>
#include <AMReX_BCUtil.H>
#include <AMReX_BC_TYPES.H>
#include <AMReX_Box.H>
#include <AMReX_MFIter.H>

using namespace amrex;

AMREX_GPU_HOST
void compute_flux_imex_o1(
    const amrex::Real time, const amrex::Geometry &geom,
    const amrex::MultiFab &statein_imp, const amrex::MultiFab &statein_exp,
    amrex::MultiFab                               &pressure,
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> &fluxes,
    const amrex::MultiFab &LS, const amrex::iMultiFab &gfm_flags,
    const amrex::Real dt, const bamrexBCData &bc_data,
    const IMEXSettings &settings)
{
    BL_PROFILE("compute_flux_imex_o1()");
    AMREX_ASSERT(pressure.nGrow() >= 1);
    AMREX_ASSERT(pressure.isDefined());
    AMREX_ASSERT(!pressure.contains_nan(0, 1, 1));
    if (settings.advection_flux == IMEXSettings::muscl_rusanov)
    {
        AMREX_ASSERT(statein_imp.nGrow() >= 3);
        AMREX_ASSERT(statein_exp.nGrow() >= 3);
        AMREX_ASSERT(!statein_imp.contains_nan(0, EULER_NCOMP, 3));
        AMREX_ASSERT(!statein_exp.contains_nan(0, EULER_NCOMP, 3));
    }
    else
    {
        AMREX_ASSERT(statein_imp.nGrow() >= 2);
        AMREX_ASSERT(statein_exp.nGrow() >= 2);
        AMREX_ASSERT(!statein_imp.contains_nan(0, EULER_NCOMP, 2));
        AMREX_ASSERT(!statein_exp.contains_nan(0, EULER_NCOMP, 2));
    }

    // Define temporary MultiFabs

    // explicitly updated values and enthalpy need 1 ghost cell
    // rhs, velocity, KE, stateout do not need ghost cells
    // note stateout is just temporary. It is not fully updated at the end of
    // the routine
    MultiFab stateex, enthalpy, rhs, velocity, ke;
    stateex.define(statein_imp.boxArray(), statein_imp.DistributionMap(),
                   EULER_NCOMP, 1);
    enthalpy.define(statein_imp.boxArray(), statein_imp.DistributionMap(), 1,
                    1);
    rhs.define(statein_imp.boxArray(), statein_imp.DistributionMap(), 1, 0);
    ke.define(statein_imp.boxArray(), statein_imp.DistributionMap(), 1, 0);
    if (settings.linearise && settings.ke_method == IMEXSettings::split)
        velocity.define(statein_imp.boxArray(), statein_imp.DistributionMap(),
                        AMREX_SPACEDIM, 0);

    const auto &dx  = geom.CellSizeArray();
    const auto &idx = geom.InvCellSizeArray();
    const Real  eps = AmrLevelAdv::h_prob_parm->epsilon;
    AMREX_D_TERM(const amrex::Real hdtdx = 0.5 * dt * idx[0];
                 , const amrex::Real hdtdy = 0.5 * dt * idx[1];
                 , const amrex::Real hdtdz = 0.5 * dt * idx[2];)
    AMREX_D_TERM(Real hdtdxe = hdtdx / eps;, Real hdtdye = hdtdy / eps;
                 , Real                           hdtdze = hdtdz / eps;)

    // First advance with Rusanov flux and compute everything needed for
    // pressure equation We do this on a box grown by 1
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    {
        for (MFIter mfi(stateex, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box &bx = mfi.tilebox();
            // boxes that well us where we're allowed to write fluxes
            GpuArray<Box, BL_SPACEDIM> nbx;
            AMREX_D_TERM(nbx[0] = mfi.nodaltilebox(0);
                         , nbx[1] = mfi.nodaltilebox(1);
                         , nbx[2] = mfi.nodaltilebox(2));
            const Box gbx = mfi.growntilebox(1);

            if (settings.advection_flux == IMEXSettings::rusanov)
                advance_rusanov_adv(time, gbx, nbx, statein_exp[mfi],
                                    statein_imp[mfi], stateex[mfi],
                                    AMREX_D_DECL(fluxes[0][mfi],
                                                 fluxes[1][mfi],
                                                 fluxes[2][mfi]),
                                    dx, dt);
            else if (settings.advection_flux == IMEXSettings::muscl_rusanov)
                advance_MUSCL_rusanov_adv(
                    time, gbx, nbx, statein_exp[mfi], statein_imp[mfi],
                    stateex[mfi],
                    AMREX_D_DECL(fluxes[0][mfi], fluxes[1][mfi],
                                 fluxes[2][mfi]),
                    dx, dt,
                    settings.butcher_tableau == IMEXButcherTableau::SP111);
            else
                Abort("Not implemented");

            compute_enthalpy(gbx, statein_exp[mfi], stateex[mfi],
                             pressure[mfi], enthalpy[mfi], settings);
            compute_ke(bx, statein_exp[mfi], stateex[mfi], statein_imp[mfi],
                       ke[mfi], settings);
            if (settings.linearise
                && settings.ke_method == IMEXSettings::split)
                compute_velocity(bx, statein_exp[mfi], velocity[mfi]);
        } // sync
        for (MFIter mfi(stateex, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box &bx = mfi.tilebox();
            pressure_source_vector(bx, stateex[mfi], enthalpy[mfi], ke[mfi],
                                   rhs[mfi],
                                   AMREX_D_DECL(hdtdx, hdtdy, hdtdz));
        }
    }

    // Fill boundary cells for pressure
    // TODO: check if this initialises pressure on course fine boundaries!
    // FillDomainBoundary(MFpressure, geom, { bcs[1 + AMREX_SPACEDIM] });
    // MFpressure.FillBoundary(geom.periodicity());
    AMREX_ASSERT(!pressure.contains_nan());

    if (settings.linearise)
    {
        solve_pressure_eqn(geom, enthalpy, velocity, rhs, pressure, stateex,
                           LS, gfm_flags, dt, bc_data, settings);
    }
    else
    {
        // Picard iteration
        // Stores result of picard iteration for momentum and density (doesn't
        // store energy) We only need enthalpy and pressure for calculating
        // fluxes so it's only defined at this scope
        // We only have an index for density so that indexing isn't confusing
        // later on - we don't actually *use* the density stored here
        MultiFab stateout(statein_imp.boxArray(),
                          statein_imp.DistributionMap(), EULER_NCOMP - 1, 0);
        // Copy explicitly updated density to out state
        stateout.ParallelCopy(stateex, 0, 0, 1, stateex.nGrow(),
                              stateout.nGrow());

        details::picard_iterate(
            geom, statein_exp, stateex, velocity, stateout, enthalpy, ke, rhs,
            pressure, LS, gfm_flags, dt, AMREX_D_DECL(hdtdx, hdtdy, hdtdz),
            AMREX_D_DECL(hdtdxe, hdtdye, hdtdze), bc_data, settings);
    }

    // Now pressure equation has been solved we can compute the fluxes
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    {
        for (MFIter mfi(stateex, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            // boxes that well us where we're allowed to write fluxes
            GpuArray<Box, BL_SPACEDIM> nbx;
            AMREX_D_TERM(nbx[0] = mfi.nodaltilebox(0);
                         , nbx[1] = mfi.nodaltilebox(1);
                         , nbx[2] = mfi.nodaltilebox(2));
            update_momentum_flux(
                nbx, pressure[mfi],
                AMREX_D_DECL(fluxes[0][mfi], fluxes[1][mfi], fluxes[2][mfi]));
            update_energy_flux(
                nbx, stateex[mfi], enthalpy[mfi], pressure[mfi],
                AMREX_D_DECL(fluxes[0][mfi], fluxes[1][mfi], fluxes[2][mfi]),
                AMREX_D_DECL(hdtdxe, hdtdye, hdtdze));
        }
    }
}

AMREX_GPU_HOST
void compute_enthalpy(const amrex::Box &bx, const amrex::FArrayBox &statein,
                      const amrex::FArrayBox &stateexp,
                      amrex::FArrayBox &a_pressure, amrex::FArrayBox &enthalpy,
                      const IMEXSettings &settings)
{
    BL_PROFILE("compute_enthalpy()");
    const Array4<const Real> in   = statein.const_array();
    const Array4<const Real> exp  = stateexp.const_array();
    const Array4<Real>       enth = enthalpy.array();

    const Real adia = AmrLevelAdv::h_prob_parm->adiabatic;
    if (!settings.linearise)
    {
        const Array4<const Real> p = a_pressure.const_array();
        ParallelFor(bx,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                        enth(i, j, k) = p(i, j, k) * adia
                                        / ((adia - 1) * exp(i, j, k, 0));
                    });
    }
    else if (settings.enthalpy_method == IMEXSettings::reuse_pressure)
    {
        const Array4<const Real> p = a_pressure.const_array();
        ParallelFor(bx,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                        enth(i, j, k) = p(i, j, k) * adia
                                        / ((adia - 1) * exp(i, j, k, 0));
                    });
    }
    else if (settings.enthalpy_method == IMEXSettings::conservative)
    {
        const Array4<Real> p = a_pressure.array();
        ParallelFor(bx,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k)
                    {
                        enth(i, j, k)
                            = vol_enthalpy(i, j, k, in) / exp(i, j, k, 0);
                        p(i, j, k) = pressure(i, j, k, exp);
                    });
    }
    else if (settings.enthalpy_method == IMEXSettings::conservative_ex)
    {
        const Array4<Real> p = a_pressure.array();
        ParallelFor(bx,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k)
                    {
                        enth(i, j, k)
                            = vol_enthalpy(i, j, k, exp) / exp(i, j, k, 0);
                        p(i, j, k) = pressure(i, j, k, exp);
                    });
    }
}

AMREX_GPU_HOST void
compute_ke(const amrex::Box &bx, const amrex::FArrayBox &statein,
           const amrex::FArrayBox &stateexp, const amrex::FArrayBox &statenew,
           amrex::FArrayBox &ke, const IMEXSettings &settings)
{
    BL_PROFILE("compute_ke()");
    const Array4<Real> kearr = ke.array();
    if (!settings.linearise)
    {
        const Array4<const Real> newarr = statenew.const_array();
        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
                    { kearr(i, j, k) = kinetic_energy(i, j, k, newarr); });
    }
    else if (settings.ke_method == IMEXSettings::conservative_kinetic)
    {
        const Array4<const Real> consv = statein.const_array();
        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
                    { kearr(i, j, k) = kinetic_energy(i, j, k, consv); });
    }
    else if (settings.ke_method == IMEXSettings::ex)
    {
        const Array4<const Real> consv_ex = stateexp.const_array();
        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
                    { kearr(i, j, k) = kinetic_energy(i, j, k, consv_ex); });
    }
    else if (settings.ke_method == IMEXSettings::split)
    {
        const Array4<const Real> consv_ex = stateexp.const_array();
        const Array4<const Real> in       = statein.const_array();
        const Real               eps      = AmrLevelAdv::h_prob_parm->epsilon;
        ParallelFor(
            bx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                // 0.5 eps u^n . q^{n+1}
                kearr(i, j, k)
                    = 0.5 * eps
                      * (AMREX_D_TERM(in(i, j, k, 1) * consv_ex(i, j, k, 1),
                                      +in(i, j, k, 2) * consv_ex(i, j, k, 2),
                                      +in(i, j, k, 3) * consv_ex(i, j, k, 3)))
                      / in(i, j, k, 0);
            });
    }
}

AMREX_GPU_HOST
void compute_velocity(const amrex::Box &bx, const amrex::FArrayBox &state,
                      amrex::FArrayBox &dst)
{
    BL_PROFILE("compute_velocity()");
    AMREX_ASSERT(state.nComp() == EULER_NCOMP);
    AMREX_ASSERT(dst.nComp() == AMREX_SPACEDIM);
    const auto &in  = state.const_array();
    const auto &out = dst.array();
    ParallelFor(
        bx, AMREX_GPU_DEVICE[=](int i, int j, int k) {
            for (int d = 0; d < AMREX_SPACEDIM; ++d)
                out(i, j, k, d) = in(i, j, k, 1 + d) / in(i, j, k, 0);
        });
}

AMREX_GPU_HOST
void compute_pressure(const amrex::MultiFab &statein_exp_new,
                      const amrex::MultiFab &statein_exp_old,
                      amrex::MultiFab &a_pressure, const amrex::Geometry &geom,
                      const amrex::Real adt, const IMEXSettings &settings)
{
    BL_PROFILE("compute_pressure()");
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    {
        const Real  eps  = AmrLevelAdv::h_prob_parm->epsilon;
        const Real  adia = AmrLevelAdv::h_prob_parm->adiabatic;
        const auto &dx   = geom.CellSizeArray();
        amrex::ignore_unused(adt, dx);
        for (MFIter mfi(a_pressure, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const auto &bx   = mfi.growntilebox();
            const auto &p    = a_pressure.array(mfi);
            const auto &sold = statein_exp_old.const_array(mfi);
            const auto &snew = statein_exp_new.const_array(mfi);
            // FI
            if (!settings.linearise)
                ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
                            { p(i, j, k) = pressure(i, j, k, snew); });
            // SK schemes
            else if (settings.linearise
                     && settings.ke_method == IMEXSettings::split)
                ParallelFor(
                    bx,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k)
                    {
                        p(i, j, k)
                            = (adia - 1)
                              * (snew(i, j, k, 1 + AMREX_SPACEDIM)
                                 - 0.5 * eps
                                       * (AMREX_D_TERM(sold(i, j, k, 1)
                                                           * snew(i, j, k, 1),
                                                       +sold(i, j, k, 2)
                                                           * snew(i, j, k, 2),
                                                       +sold(i, j, k, 3)
                                                           * snew(i, j, k, 3)))
                                       / sold(i, j, k, 0));
                    });
            // EK schemes
            else if (settings.linearise
                     && settings.ke_method == IMEXSettings::ex)
            {
                // Explicitly updated as if doing update from (U_E)^{l-1} to
                // (U_E)^l
                FArrayBox consv_exp(bx, EULER_NCOMP, The_Async_Arena());
                const Array4<const Real>     &ex = consv_exp.const_array();
                FArrayBox                     dummy;
                GpuArray<Box, AMREX_SPACEDIM> dummy_bx;
                if (settings.advection_flux == IMEXSettings::rusanov)
                    advance_rusanov_adv(0, bx, dummy_bx, statein_exp_old[mfi],
                                        statein_exp_old[mfi], consv_exp,
                                        AMREX_D_DECL(dummy, dummy, dummy), dx,
                                        adt);
                else if (settings.advection_flux
                         == IMEXSettings::muscl_rusanov)
                    advance_MUSCL_rusanov_adv(
                        0, bx, dummy_bx, statein_exp_old[mfi],
                        statein_exp_old[mfi], consv_exp,
                        AMREX_D_DECL(dummy, dummy, dummy), dx, adt);
                else
                    Abort("Not implemented");

                // Now compute pressure
                ParallelFor(
                    bx,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k)
                    {
                        p(i, j, k)
                            = (adia - 1)
                              * (snew(i, j, k, 1 + AMREX_SPACEDIM)
                                 - 0.5 * eps
                                       * (AMREX_D_TERM(
                                           ex(i, j, k, 1) * ex(i, j, k, 1),
                                           +ex(i, j, k, 2) * ex(i, j, k, 2),
                                           +ex(i, j, k, 3) * ex(i, j, k, 3)))
                                       / snew(i, j, k, 0));
                    });
            }
            else // any other methods
                ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
                            { p(i, j, k) = pressure(i, j, k, snew); });
        }
    }
}

AMREX_GPU_HOST
void solve_pressure_eqn(const amrex::Geometry &geom,
                        const amrex::MultiFab &enthalpy,
                        const amrex::MultiFab &velocity,
                        const amrex::MultiFab &rhs, amrex::MultiFab &pressure,
                        const amrex::MultiFab &U, const amrex::MultiFab &LS,
                        const amrex::iMultiFab &gfm_flags,
                        const amrex::Real dt, const bamrexBCData &bc_data,
                        const IMEXSettings &settings)
{
    AMREX_ASSERT(rhs.nComp() == 1);
    AMREX_ASSERT(pressure.nComp() == 1);
    // If rigid bodies are disabled
    // An operator representing an equation of the form
    // (A * alpha(x) - B div (beta(x) grad) + C c . grad) phi = f
    // If rigid bodies enabled
    // An operator representing an equation of the form
    // (A * alpha(x) - B div (beta(x) grad)) phi = f
    //  (with support for embedded boundaries)

    // Ownership belongs to this scope
    // There are probably ASync issues with using this if on GPU
    auto pressure_op = details::get_p_linop(bc_data.rb_enabled());

    if (!bc_data.rb_enabled())
        details::setup_pressure_op(
            geom, enthalpy, velocity, rhs, pressure,
            dynamic_cast<MLABecLaplacianGrad &>(*pressure_op), U, dt, bc_data,
            settings);
    else
    {
#ifdef AMREX_USE_EB
        IMEXEB::setup_pressure_op_rb(geom, enthalpy, rhs, pressure,
                                     dynamic_cast<MLEBABecLap &>(*pressure_op),
                                     U, dt, bc_data);
#endif
    }

    MLMG solver(*pressure_op);

    details::solve_pressure(geom, rhs, pressure, U, LS, gfm_flags, solver,
                            bc_data, settings);
}

AMREX_GPU_HOST
void pressure_source_vector(const amrex::Box       &bx,
                            const amrex::FArrayBox &stateexp,
                            const amrex::FArrayBox &enthalpy,
                            const amrex::FArrayBox &ke, amrex::FArrayBox &dst,
                            AMREX_D_DECL(amrex::Real hdtdx, amrex::Real hdtdy,
                                         amrex::Real hdtdz))
{
    BL_PROFILE("pressure_source_vector()");
    const Real               epsilon  = AmrLevelAdv::h_prob_parm->epsilon;
    const Array4<const Real> consv_ex = stateexp.const_array();
    const Array4<const Real> enth     = enthalpy.const_array();
    const Array4<const Real> ke_arr   = ke.const_array();
    const Array4<Real>       b        = dst.array();

    ParallelFor(
        bx,
        [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            b(i, j, k)
                = epsilon
                  * (consv_ex(i, j, k, 1 + AMREX_SPACEDIM) - ke_arr(i, j, k)
                     - (AMREX_D_TERM(
                         hdtdx
                             * (enth(i + 1, j, k) * consv_ex(i + 1, j, k, 1)
                                - enth(i - 1, j, k)
                                      * consv_ex(i - 1, j, k, 1)),
                         +hdtdy
                             * (enth(i, j + 1, k) * consv_ex(i, j + 1, k, 2)
                                - enth(i, j - 1, k)
                                      * consv_ex(i, j - 1, k, 2)),
                         +hdtdz
                             * (enth(i, j, k + 1) * consv_ex(i, j, k + 1, 3)
                                - enth(i, j, k - 1)
                                      * consv_ex(i, j, k - 1, 3)))));
        });
}

AMREX_GPU_HOST
void update_momentum(const amrex::Box &bx, const amrex::FArrayBox &stateex,
                     const amrex::FArrayBox &pressure, amrex::FArrayBox &dst,
                     AMREX_D_DECL(amrex::Real hdtdxe, amrex::Real hdtdye,
                                  amrex::Real hdtdze))
{
    BL_PROFILE("update_momentum()");
    const auto &out = dst.array();
    const auto &p   = pressure.const_array();
    const auto &ex  = stateex.const_array();
    ParallelFor(
        bx,
        [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            AMREX_D_TERM(
                out(i, j, k, 1)
                = ex(i, j, k, 1) - hdtdxe * (p(i + 1, j, k) - p(i - 1, j, k));
                ,
                out(i, j, k, 2)
                = ex(i, j, k, 2) - hdtdye * (p(i, j + 1, k) - p(i, j - 1, k));
                ,
                out(i, j, k, 3)
                = ex(i, j, k, 3) - hdtdze * (p(i, j, k + 1) - p(i, j, k - 1));)
        });
}

/**
 * \brief Updates momentum flux, given fluxes from explicit update and solution
 * to pressure equation
 */
AMREX_GPU_HOST
void update_momentum_flux(
    const amrex::GpuArray<amrex::Box, AMREX_SPACEDIM> &nbx,
    const amrex::FArrayBox                            &pressure,
    AMREX_D_DECL(amrex::FArrayBox &fx, amrex::FArrayBox &fy,
                 amrex::FArrayBox &fz))
{
    BL_PROFILE("update_momentum_flux()");
    AMREX_D_TERM(const auto &fluxx = fx.array();
                 , const auto &fluxy = fy.array();
                 , const auto &fluxz = fz.array();)
    const auto &p    = pressure.const_array();
    const Real  heps = 0.5 / AmrLevelAdv::h_prob_parm->epsilon;
    ParallelFor(
        AMREX_D_DECL(nbx[0], nbx[1], nbx[2]),
        AMREX_D_DECL(
            [=] AMREX_GPU_DEVICE(int i, int j, int k)
            { fluxx(i, j, k, 1) += heps * (p(i, j, k) + p(i - 1, j, k)); },
            [=] AMREX_GPU_DEVICE(int i, int j, int k)
            { fluxy(i, j, k, 2) += heps * (p(i, j, k) + p(i, j - 1, k)); },
            [=] AMREX_GPU_DEVICE(int i, int j, int k)
            { fluxz(i, j, k, 3) += heps * (p(i, j, k) + p(i, j, k - 1)); }));
}

/**
 * Updates energy and computes flux. Assumes momentum and density are already
 * updated in \c dst
 */
AMREX_GPU_HOST
void update_energy_flux(
    amrex::GpuArray<amrex::Box, AMREX_SPACEDIM> nbx,
    const amrex::FArrayBox &stateex, const amrex::FArrayBox &enthalpy,
    const amrex::FArrayBox &pressure,
    AMREX_D_DECL(amrex::FArrayBox &fx, amrex::FArrayBox &fy,
                 amrex::FArrayBox &fz),
    AMREX_D_DECL(amrex::Real hdtdxe, amrex::Real hdtdye, amrex::Real hdtdze))
{
    BL_PROFILE("update_energy_flux()");
    const auto &p    = pressure.const_array();
    const auto &ex   = stateex.const_array();
    const auto &enth = enthalpy.const_array();

    // Conservative flux is calculated in the same way for all methods, the
    // difference being that we will have updated enthalpy and KE before
    // calling this whereas if linearised they should be the same as when used
    // in the pressure equation
    AMREX_D_TERM(const auto &fluxx = fx.array();
                 , const auto &fluxy = fy.array();
                 , const auto &fluxz = fz.array();)
    ParallelFor(AMREX_D_DECL(nbx[0], nbx[1], nbx[2]),
                AMREX_D_DECL(
                    [=] AMREX_GPU_DEVICE(int i, int j, int k)
                    {
                        fluxx(i, j, k, 1 + AMREX_SPACEDIM)
                            += 0.5
                                   * (enth(i, j, k) * ex(i, j, k, 1)
                                      + enth(i - 1, j, k) * ex(i - 1, j, k, 1))
                               - hdtdxe
                                     * ((enth(i, j, k) + enth(i - 1, j, k))
                                        * (p(i, j, k) - p(i - 1, j, k)));
                    },
                    [=] AMREX_GPU_DEVICE(int i, int j, int k)
                    {
                        fluxy(i, j, k, 1 + AMREX_SPACEDIM)
                            += 0.5
                                   * (enth(i, j, k) * ex(i, j, k, 2)
                                      + enth(i, j - 1, k) * ex(i, j - 1, k, 2))
                               - hdtdye
                                     * ((enth(i, j, k) + enth(i, j - 1, k))
                                        * (p(i, j, k) - p(i, j - 1, k)));
                    },
                    [=] AMREX_GPU_DEVICE(int i, int j, int k)
                    {
                        fluxz(i, j, k, 1 + AMREX_SPACEDIM)
                            += 0.5
                                   * (enth(i, j, k) * ex(i, j, k, 3)
                                      + enth(i, j, k - 1) * ex(i, j, k - 1, 3))
                               - hdtdze
                                     * ((enth(i, j, k) + enth(i, j, k - 1))
                                        * (p(i, j, k) - p(i, j, k - 1)));
                    }));
}

namespace details
{

AMREX_GPU_HOST
void setup_pressure_op(
    const amrex::Geometry &geom, const amrex::MultiFab &enthalpy,
    const amrex::MultiFab &velocity, const amrex::MultiFab &rhs,
    amrex::MultiFab &pressure_scratch, amrex::MLABecLaplacianGrad &pressure_op,
    const amrex::MultiFab &U, const amrex::Real dt,
    const bamrexBCData &bc_data, const IMEXSettings &settings)
{
    BL_PROFILE("details::setup_pressure_op()");
    const bool use_grad_term
        = (settings.linearise && settings.ke_method == IMEXSettings::split);
    AMREX_ASSERT(enthalpy.nComp() == 1);
    AMREX_ASSERT(velocity.nComp() == AMREX_SPACEDIM || !use_grad_term);

    pressure_op.define({ geom }, { rhs.boxArray() },
                       { rhs.DistributionMap() });

    // BCs
    details::set_pressure_domain_BC(pressure_op, geom, pressure_scratch, U,
                                    bc_data);

    // set coefficients
    const amrex::Real eps        = AmrLevelAdv::h_prob_parm->epsilon;
    const amrex::Real adia       = AmrLevelAdv::h_prob_parm->adiabatic;
    const amrex::Real gradscalar = use_grad_term ? -0.5 * eps * dt : 0;
    pressure_op.setScalars(1, dt * dt, gradscalar);
    pressure_op.setACoeffs(0, eps / (adia - 1));

    // Construct face-centred enthalpy
    Array<MultiFab, AMREX_SPACEDIM> enthalpy_f;
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        BoxArray ba = rhs.boxArray();
        ba.surroundingNodes(d);
        enthalpy_f[d].define(ba, rhs.DistributionMap(), 1, 0);
    }
    amrex::average_cellcenter_to_face(GetArrOfPtrs(enthalpy_f), enthalpy,
                                      geom);

    pressure_op.setBCoeffs(0, amrex::GetArrOfConstPtrs(enthalpy_f));
    if (use_grad_term)
    {
        pressure_op.setCCoeffs(0, velocity);
    }
}

AMREX_GPU_HOST
void solve_pressure(const amrex::Geometry &geom, const amrex::MultiFab &rhs,
                    amrex::MultiFab &pressure, const amrex::MultiFab &U,
                    const amrex::MultiFab  &LS,
                    const amrex::iMultiFab &gfm_flags, amrex::MLMG &solver,
                    const bamrexBCData &bc_data, const IMEXSettings &settings)
{
    BL_PROFILE("details::solve_pressure()");
    AMREX_ASSERT(pressure.nComp() == 1);
    AMREX_ASSERT(rhs.nComp() == 1);
    // Configure solver
    solver.setMaxIter(1000);
    solver.setMaxFmgIter(0);
    solver.setVerbose(settings.solver_verbose);
    solver.setBottomVerbose(settings.bottom_solver_verbose);

    const Real TOL_RES = 1e-12;
    const Real TOL_ABS = 1e-15;

    // TODO: use hypre here
    solver.solve({ &pressure }, { &rhs }, TOL_RES, TOL_ABS);

    bc_data.fill_pressure_boundary(geom, pressure, U, 0);
    // Extrapolate pressure into the ghost cells
    if (bc_data.rb_enabled())
    {
        AMREX_ASSERT(bc_data.rigidbody_bc() == RigidBodyBCType::reflective
                     || bc_data.rigidbody_bc() == RigidBodyBCType::no_slip);
        fill_ghost_p_rb(geom, pressure, LS, gfm_flags, bc_data.rigidbody_bc());
    }

    AMREX_ASSERT(!pressure.contains_nan());
}

AMREX_GPU_HOST
void copy_pressure(const amrex::MultiFab &statesrc, amrex::MultiFab &dst)
{
    BL_PROFILE("copy_pressure()");
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    {
        for (MFIter mfi(dst, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box               &bx      = mfi.tilebox();
            const Array4<const Real> src     = statesrc.array(mfi);
            const Array4<Real>       dst_arr = dst.array(mfi);
            ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
                        { dst_arr(i, j, k) = pressure(i, j, k, src); });
        }
    }
}

AMREX_GPU_HOST
void set_pressure_domain_BC(amrex::MLLinOp        &pressure_op,
                            const amrex::Geometry &geom,
                            amrex::MultiFab       &pressure_scratch,
                            const amrex::MultiFab &U,
                            const bamrexBCData    &bc_data)
{
    const auto &lobc = bc_data.get_p_linop_lobc();
    const auto &hibc = bc_data.get_p_linop_hibc();
    pressure_op.setDomainBC(lobc, hibc);

    bool dirichlet = false;
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
    {
        if (lobc[dir] == LinOpBCType::Dirichlet
            || hibc[dir] == LinOpBCType::Dirichlet)
            dirichlet = true;
        if (lobc[dir] == LinOpBCType::Robin || hibc[dir] == LinOpBCType::Robin)
            Abort("Robin BCs for pressure linear solve not implemented");
        if (lobc[dir] == LinOpBCType::inhomogNeumann
            || hibc[dir] == LinOpBCType::inhomogNeumann)
            Abort("Inhomogenous Neumann BCs for pressure linear solve not "
                  "implemented");
    }

    // if (level > 0)
    //     pressure_op.setCoarseFineBC(&crse_pressure, crse_ratio[0]);

    if (!dirichlet)
        // pure homogeneous Neumann BC so we pass nullptr
        pressure_op.setLevelBC(0, nullptr);
    else
    {
        // Fill dirichlet conditions
        bc_data.fill_pressure_boundary(geom, pressure_scratch, U, 0);
        pressure_op.setLevelBC(0, &pressure_scratch);
    }
}

AMREX_GPU_HOST
void picard_iterate(
    const amrex::Geometry &geom, const amrex::MultiFab &statein,
    const amrex::MultiFab &stateex, const amrex::MultiFab &velocity,
    amrex::MultiFab &stateout, amrex::MultiFab &enthalpy, amrex::MultiFab &ke,
    amrex::MultiFab &rhs, amrex::MultiFab &pressure, const amrex::MultiFab &LS,
    const amrex::iMultiFab &gfm_flags, amrex::Real dt,
    AMREX_D_DECL(amrex::Real hdtdx, amrex::Real hdtdy, amrex::Real hdtdz),
    AMREX_D_DECL(amrex::Real hdtdxe, amrex::Real hdtdye, amrex::Real hdtdze),
    const bamrexBCData &bc_data, const IMEXSettings &settings)
{
    BL_PROFILE("picard_iterate()");
    MultiFab residual;
    if (settings.compute_residual)
        residual.define(pressure.boxArray(), pressure.DistributionMap(), 1, 0);

    for (int iter = 0; iter < settings.picard_max_iter; ++iter)
    {
        if (settings.verbose)
            Print() << "[IMEX] Beginning Picard iteration " << iter << "..."
                    << std::endl;

        // here we call the other functions in details:: so that we can compute
        // the residual
        {
            auto pressure_op = details::get_p_linop(bc_data.rb_enabled());

            if (!bc_data.rb_enabled())
                details::setup_pressure_op(
                    geom, enthalpy, velocity, rhs, pressure,
                    dynamic_cast<MLABecLaplacianGrad &>(*pressure_op), stateex,
                    dt, bc_data, settings);
            else
            {
#ifdef AMREX_USE_EB
                IMEXEB::setup_pressure_op_rb(
                    geom, enthalpy, rhs, pressure,
                    dynamic_cast<MLEBABecLap &>(*pressure_op), stateex, dt,
                    bc_data);
#endif
            }

            // solver can only be constructed once the operator is initialised
            MLMG residual_solver(*pressure_op);
            MLMG solver(*pressure_op);

            if (settings.compute_residual)
            {
                residual_solver.compResidual({ &residual }, { &pressure },
                                             { &rhs });
                const Real res_norm = residual.norm2();
                const Real res_rel  = res_norm / rhs.norm2();
                if (settings.verbose)
                    Print() << "\tInitial residual = " << res_norm
                            << "\tresidual/bnorm = " << res_rel << std::endl;
                if (fabs(res_rel) < settings.res_tol)
                {
                    if (settings.verbose)
                        Print()
                            << "Picard iteration reached convergence after "
                            << iter << " iterations" << std::endl;
                    return;
                }
            }

            details::solve_pressure(geom, rhs, pressure, stateex, LS,
                                    gfm_flags, solver, bc_data, settings);
        }

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        {
            for (MFIter mfi(stateout, TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                const Box &bx  = mfi.tilebox();
                const Box  gbx = mfi.growntilebox(1);
                update_momentum(bx, stateex[mfi], pressure[mfi], stateout[mfi],
                                AMREX_D_DECL(hdtdxe, hdtdye, hdtdze));
                // we don't need to pass the new state, because the density in
                // stateex is the new density already, and we use the pressure
                // and density to calculate enthalpy
                compute_enthalpy(gbx, statein[mfi], stateex[mfi],
                                 pressure[mfi], enthalpy[mfi], settings);
                compute_ke(bx, statein[mfi], stateex[mfi], stateout[mfi],
                           ke[mfi], settings);
            } // sync
            for (MFIter mfi(stateex, TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                const Box &bx = mfi.tilebox();
                pressure_source_vector(bx, stateex[mfi], enthalpy[mfi],
                                       ke[mfi], rhs[mfi],
                                       AMREX_D_DECL(hdtdx, hdtdy, hdtdz));
            }
        }
    }

    if (settings.verbose)
        Print() << "[IMEX] Reached maximum number of Picard iterations = "
                << settings.picard_max_iter << "!" << std::endl;
}

} // namespace details