/**
 * Boscheri and Pareshi formulation
 */
#include "IMEX/IMEX_O1_bp.H"
#include "AmrLevelAdv.H"
#include "IMEX/IMEX_O1.H"
#include "IMEX_K/euler_K.H"
#include "LinearSolver/MLALaplacianGrad.H"
#include "LinearSolver/MLALaplacianGradAlt.H"
#include "MFIterLoop.H"

#include <AMReX_BCUtil.H>
#include <AMReX_BC_TYPES.H>
#include <AMReX_Box.H>
#include <AMReX_MLMG.H>
#include <AMReX_PlotFileUtil.H>

using namespace amrex;

/**
 * Boscheri and Pareshi imex update
 */
AMREX_GPU_HOST
void advance_o1_pimex_bp(int level, amrex::IntVect &crse_ratio,
                         const amrex::Real time, const amrex::Geometry &geom,
                         const amrex::MultiFab &statein,
                         amrex::MultiFab       &stateout,
                         amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> &fluxes,
                         const amrex::MultiFab             *crse_soln,
                         const amrex::Real                  dt,
                         const amrex::Vector<amrex::BCRec> &bcs, int verbose,
                         int bottom_verbose)
{
    /**
     * This formulation follows Allegrini 2023 Order 1 AP L^2 scheme for
     * pressure eqn and Rusanov flux for explicit update
     */
    AMREX_ASSERT(statein.nGrow() >= 2);
#ifdef AMREX_USE_GPU
    // AMREX_ASSERT(fluxes[0].nGrow() >= 1);
#endif

    // Define temporary MultiFabs
    BL_PROFILE_VAR("advance_o1_pimex() allocation", palloc);

    // MFex, MFpressure, and MFscaledenth_cc have ghost cells
    // MFex and MFscaledenth_cc have 1 set of ghost cells,
    // MFpressure has 2

    const int NSTATE = AmrLevelAdv::NUM_STATE;
    MultiFab  MFex, MFb, MFscaledenth_cc, MFpgradcoef;
    Array<MultiFab, AMREX_SPACEDIM> MFscaledenth_f;
    MFex.define(stateout.boxArray(), stateout.DistributionMap(), NSTATE, 1);
    MFb.define(stateout.boxArray(), stateout.DistributionMap(), 1, 0);
    MFscaledenth_cc.define(stateout.boxArray(), stateout.DistributionMap(), 1,
                           1);
    MFpgradcoef.define(stateout.boxArray(), stateout.DistributionMap(),
                       AMREX_SPACEDIM, 0);

#if REUSE_PRESSURE
    MultiFab& MFpressure = AmrLevelAdv::MFpressure;
    bool p_needs_update = false;
    if (!MFpressure.isDefined())
    {
        p_needs_update = true;
        MFpressure.define(stateout.boxArray(), stateout.DistributionMap(), 1, 2);
    }
#else
    MultiFab MFpressure;
    MFpressure.define(stateout.boxArray(), stateout.DistributionMap(), 1, 2);
#endif

    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        BoxArray ba = stateout.boxArray();
        ba.surroundingNodes(d);
        MFscaledenth_f[d].define(ba, stateout.DistributionMap(), 1, 0);
    }

    MultiFab crse_pressure;
    if (level > 0)
    {
        AMREX_ASSERT(crse_soln);
        for (int d = 1; d < AMREX_SPACEDIM; ++d)
        {
            AMREX_ASSERT(crse_ratio[d] == crse_ratio[0]);
        }
        crse_pressure.define(crse_soln->boxArray(),
                             crse_soln->DistributionMap(), crse_soln->nComp(),
                             0);
        copy_pressure(*crse_soln, crse_pressure);
    }

    BL_PROFILE_VAR_STOP(palloc);
    BL_PROFILE_VAR("advance_o1_pimex() explicit and enthalpy", pexp);

    const Real                  epsilon = AmrLevelAdv::h_prob_parm->epsilon;
    const Real                  adia    = AmrLevelAdv::h_prob_parm->adiabatic;
    GpuArray<Real, BL_SPACEDIM> dx      = geom.CellSizeArray();

    // First advance with Rusanov flux and compute scaled enthalpy
    // We do this on a box grown by 1 (but without overlapping between tiles)
    MFIter_loop(
        time, geom, statein, MFex, fluxes, dt,
        [&](const amrex::MFIter &mfi, const amrex::Real time,
            const amrex::Box                           &bx,
            amrex::GpuArray<amrex::Box, AMREX_SPACEDIM> nbx,
            const amrex::FArrayBox &statein, amrex::FArrayBox &consv_ex,
            AMREX_D_DECL(amrex::FArrayBox & fx, amrex::FArrayBox & fy,
                         amrex::FArrayBox & fz),
            amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx,
            const amrex::Real                            dt)
        {
            const Box gbx = mfi.growntilebox(1);
            advance_rusanov_adv(time, gbx, nbx, statein, consv_ex,
                                AMREX_D_DECL(fx, fy, fz), dx, dt);
            // consv_ex is now U^{n+1, ex}

            // now construct cell centred scaled enthalpies
            // they are h^{n+1, ex} / density^{n+1, ex}
            const Array4<Real> scaledenth_cc      = MFscaledenth_cc.array(mfi);
            const Array4<const Real> consv_ex_arr = consv_ex.array();
            const Array4<Real>       p            = MFpressure.array(mfi);
            const auto              &in           = statein.array();
            ParallelFor(gbx,
                        [=] AMREX_GPU_DEVICE(int i, int j, int k)
                        {
#if REUSE_PRESSURE
                            if (p_needs_update) p(i, j, k) = pressure(i, j, k, consv_ex_arr);
                            scaledenth_cc(i, j, k)
                                = p(i,j,k) * adia
                                  / ((adia - 1) *consv_ex_arr(i, j, k, 0));
#else
#   if USE_EXPLICIT_ENTH
                            scaledenth_cc(i, j, k)
                                = vol_enthalpy(i, j, k, consv_ex_arr)
                                  / consv_ex_arr(i, j, k, 0);
#   else
                            scaledenth_cc(i, j, k)
                                = vol_enthalpy(i, j, k, in)
                                  / consv_ex_arr(i, j, k, 0);
#   endif

                            // initial guess
                            // explicitly updated values only exist on a box
                            // grown by 1, so we will need to use
                            // MultiFab::FillBoundary and FillDomainBoundary to
                            // fill the second layer of ghost cells
                            p(i, j, k) = pressure(i, j, k, consv_ex_arr);
#endif
                        });

            // compute coefficients of grad p term
            const auto &pgradcoeff = MFpgradcoef.array(mfi);
            ParallelFor(bx,
                        [=] AMREX_GPU_DEVICE(int i, int j, int k)
                        {
                            // we set C = -eps * dt / 2 so the coeff is just q
                            // / rho
                            AMREX_D_TERM(pgradcoeff(i, j, k, 0)
                                         = in(i, j, k, 1) / in(i, j, k, 0);
                                         , pgradcoeff(i, j, k, 1)
                                           = in(i, j, k, 2) / in(i, j, k, 0);
                                         , pgradcoeff(i, j, k, 2)
                                           = in(i, j, k, 3) / in(i, j, k, 0));
                        });
        });

    // Fill boundary cells for pressure
    // TODO: check if this initialises pressure on course fine boundaries!
    MFpressure.FillBoundary(geom.periodicity());
    FillDomainBoundary(MFpressure, geom, { bcs[1 + AMREX_SPACEDIM] });

    BL_PROFILE_VAR_STOP(pexp);
    BL_PROFILE_VAR("advance_o1_pimex() assembly", passembly);

    // Assembly of source vector and setting coefficients

    // Calculate source vector (cannot be done in MFIter loop above because it
    // requires offset values for the explicitly updated momentum densities)
    AMREX_D_TERM(const Real hedtdx = 0.5 * epsilon * dt / dx[0];
                 , const Real hedtdy = 0.5 * epsilon * dt / dx[1];
                 , const Real hedtdz = 0.5 * epsilon * dt / dx[2];)
    const Real hedt = 0.5 * epsilon * dt;
    pressure_source_vector_bp(MFex, MFscaledenth_cc, MFpgradcoef, MFb, hedt,
                              AMREX_D_DECL(hedtdx, hedtdy, hedtdz));

    // An operator representing an equation of the form
    // (A * alpha(x) - B div (beta(x) grad) + C c . grad) phi = f
    // MLABecLaplacianGrad pressure_op({ geom }, { stateout.boxArray() },
    //                                 { stateout.DistributionMap() });
#if USE_ALT_LAPLACIAN
    MLABecLaplacianGradAlt pressure_op({ geom }, { stateout.boxArray() },
                                    { stateout.DistributionMap() }, LPInfo().setMaxCoarseningLevel(0));
#else
    MLABecLaplacianGrad pressure_op({ geom }, { stateout.boxArray() },
                                    { stateout.DistributionMap() }, LPInfo().setMaxCoarseningLevel(0));
#endif

    // BCs
    details::set_pressure_domain_BC(pressure_op, geom, bcs);

    if (level > 0)
        pressure_op.setCoarseFineBC(&crse_pressure, crse_ratio[0]);

    // pure homogeneous Neumann BC so we pass nullptr
    pressure_op.setLevelBC(0, nullptr);

    // set coefficients

    pressure_op.setScalars(1, dt * dt, -hedt);
    pressure_op.setACoeffs(0, epsilon / (adia - 1));
#if USE_ALT_LAPLACIAN
    pressure_op.setBCoeffs(0, MFscaledenth_cc);
#else
    amrex::average_cellcenter_to_face(GetArrOfPtrs(MFscaledenth_f),
                                      MFscaledenth_cc, geom);
    pressure_op.setBCoeffs(0, amrex::GetArrOfConstPtrs(MFscaledenth_f));
#endif
    pressure_op.setCCoeffs(0, MFpgradcoef);

    BL_PROFILE_VAR_STOP(passembly);
    BL_PROFILE_VAR("advance_o1_pimex() solving pressure", ppressure);

    MLMG solver(pressure_op);
    solver.setMaxIter(100);
    solver.setMaxFmgIter(0);
    solver.setVerbose(verbose);
    solver.setBottomVerbose(bottom_verbose);

    const Real TOL_RES = 1e-12;
    const Real TOL_ABS = 0;

    AMREX_ASSERT(!MFscaledenth_cc.contains_nan());
    AMREX_ASSERT(!MFex.contains_nan());
    AMREX_ASSERT(!statein.contains_nan());
    AMREX_ASSERT(!MFb.contains_nan());
    // AMREX_ASSERT(!MFscaledenth_f[0].contains_nan());
    AMREX_ASSERT(!MFpressure.contains_nan());
    AMREX_ASSERT(!MFpgradcoef.contains_nan());

    // TODO: use hypre here
    solver.solve({ &MFpressure }, { &MFb }, TOL_RES, TOL_ABS);

    AMREX_ASSERT(!MFpressure.contains_nan());

    // TODO: fill course fine boundary cells for pressure!
    MFpressure.FillBoundary(geom.periodicity());
    FillDomainBoundary(MFpressure, geom, { bcs[1 + AMREX_SPACEDIM] });

    BL_PROFILE_VAR_STOP(ppressure);
    BL_PROFILE_VAR("advance_o1_pimex() updating", pupdate);

    amrex::WriteSingleLevelPlotfile("output/tmp/pressure", MFpressure,
                                    { "pressure" }, geom, time, 0);
    amrex::WriteSingleLevelPlotfile("output/tmp/explicit", MFex,
                                    { "density", "mom_x", "energy" }, geom,
                                    time, 0);
    update_ex_from_pressure_bp(time, geom, statein, MFpressure, MFex,
                               MFscaledenth_cc, stateout, fluxes, dt);

    BL_PROFILE_VAR_STOP(pupdate);
}

/**
 * MFscaledenth_cc should be (h / rho)^n
 * MFex should be explicitly updated values
 * MFpgradcoef should be (q / rho)^n
 * hedt should be epsilon * dt / 2
 */
AMREX_GPU_HOST
void pressure_source_vector_bp(
    const amrex::MultiFab &MFex, const amrex::MultiFab &MFscaledenth_cc,
    const amrex::MultiFab &MFpgradcoef, amrex::MultiFab &dst,
    amrex::Real /*hedt*/,
    AMREX_D_DECL(amrex::Real hedtdx, amrex::Real hedtdy, amrex::Real hedtdz))
{
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    {
        const Real epsilon = AmrLevelAdv::h_prob_parm->epsilon;

        for (MFIter mfi(dst, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box               &bx        = mfi.tilebox();
            const Array4<Real>       b         = dst.array(mfi);
            const Array4<const Real> consv_ex  = MFex.array(mfi);
            const Array4<const Real> sh_cc     = MFscaledenth_cc.array(mfi);
            const Array4<const Real> pgradcoef = MFpgradcoef.array(mfi);

            ParallelFor(
                bx,
                [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    b(i, j, k)
                        = epsilon
                              * (consv_ex(i, j, k, 1 + AMREX_SPACEDIM)
                                 - 0.5 * epsilon
                                       * (AMREX_D_TERM(
                                           pgradcoef(i, j, k, 0)
                                               * consv_ex(i, j, k, 1),
                                           +pgradcoef(i, j, k, 1)
                                               * consv_ex(i, j, k, 2),
                                           +pgradcoef(i, j, k, 2)
                                               * consv_ex(i, j, k, 3))))
                          - (AMREX_D_TERM(
                              hedtdx
                                  * (sh_cc(i + 1, j, k)
                                         * consv_ex(i + 1, j, k, 1)
                                     - sh_cc(i - 1, j, k)
                                           * consv_ex(i - 1, j, k, 1)),
                              +hedtdy
                                  * (sh_cc(i, j + 1, k)
                                         * consv_ex(i, j + 1, k, 2)
                                     - sh_cc(i, j - 1, k)
                                           * consv_ex(i, j - 1, k, 2)),
                              +hedtdz
                                  * (sh_cc(i, j, k + 1)
                                         * consv_ex(i, j, k + 1, 3)
                                     - sh_cc(i, j, k - 1)
                                           * consv_ex(i, j, k - 1, 3))));
                });
        }
    }
}

AMREX_GPU_HOST
void update_ex_from_pressure_bp(
    amrex::Real /*time*/, const amrex::Geometry &geom,
    const amrex::MultiFab &statein, const amrex::MultiFab &MFpressure,
    const amrex::MultiFab &MFex, const amrex::MultiFab &MFscaledenth_cc,
    amrex::MultiFab                               &stateout,
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> &fluxes, amrex::Real dt)
{
    amrex::ignore_unused(MFscaledenth_cc, fluxes);

    const Real  adia = AmrLevelAdv::h_prob_parm->adiabatic;
    const Real  eps  = AmrLevelAdv::h_prob_parm->epsilon;
    const auto &dx   = geom.CellSizeArray();

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    {
        AMREX_D_TERM(const Real hdtedx = 0.5 * dt / (eps * dx[0]);
                     , const Real hdtedy = 0.5 * dt / (eps * dx[1]);
                     , const Real hdtedz = 0.5 * dt / (eps * dx[2]));
        for (MFIter mfi(stateout, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const auto &bx  = mfi.tilebox();
            const auto &in  = statein.const_array(mfi);
            const auto &ex  = MFex.const_array(mfi);
            const auto &p   = MFpressure.const_array(mfi);
            const auto &out = stateout.array(mfi);

            ParallelFor(
                bx,
                [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    // density
                    out(i, j, k, 0) = ex(i, j, k, 0);

                    // momentum
                    AMREX_D_TERM(
                        out(i, j, k, 1)
                        = ex(i, j, k, 1)
                          - hdtedx * (p(i + 1, j, k) - p(i - 1, j, k));
                        , out(i, j, k, 2)
                          = ex(i, j, k, 2)
                            - hdtedy * (p(i, j + 1, k) - p(i, j - 1, k));
                        , out(i, j, k, 3)
                          = ex(i, j, k, 3)
                            - hdtedz * (p(i, j, k + 1) - p(i, j, k - 1));)

                    // energy
                    out(i, j, k, 1 + AMREX_SPACEDIM)
                        = p(i, j, k) / (adia - 1)
                          + 0.5 * eps / in(i, j, k, 0)
                                * (AMREX_D_TERM(
                                    in(i, j, k, 1) * out(i, j, k, 1),
                                    +in(i, j, k, 2) * out(i, j, k, 2),
                                    +in(i, j, k, 3) * out(i, j, k, 3)));
                });
        }
    }
}
