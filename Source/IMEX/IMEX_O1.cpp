#include "IMEX/IMEX_O1.H"
#include "AmrLevelAdv.H"
#include "IMEX_K/euler_K.H"
#include "MFIterLoop.H"

#include <AMReX_BCUtil.H>
#include <AMReX_BC_TYPES.H>
#include <AMReX_Box.H>
#include <AMReX_MLABecLaplacian.H>
#include <AMReX_MLMG.H>

using namespace amrex;

AMREX_GPU_HOST
void advance_o1_pimex(int level, amrex::IntVect &crse_ratio,
                      const amrex::Real time, const amrex::Geometry &geom,
                      const amrex::MultiFab                         &statein,
                      amrex::MultiFab                               &stateout,
                      amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> &fluxes,
                      const amrex::MultiFab *crse_soln, const amrex::Real dt,
                      int verbose, int bottom_verbose)
{
    /**
     * This formulation follows Allegrini 2023 Order 1 AP L^2 scheme for
     * pressure eqn and Rusanov flux for explicit update
     */
    AMREX_ASSERT(statein.nGrow() >= 2);
#ifdef AMREX_USE_GPU
    AMREX_ASSERT(fluxes[0].nGrow() >= 1);
#endif

    // Define temporary MultiFabs
    BL_PROFILE_VAR("advance_o1_pimex() allocation", palloc);

    const int                       NSTATE = AmrLevelAdv::NUM_STATE;
    MultiFab                        MFex, MFpressure, MFb, MFscaledenth_cc;
    Array<MultiFab, AMREX_SPACEDIM> MFscaledenth_f;
    MFex.define(stateout.boxArray(), stateout.DistributionMap(), NSTATE, 1);
    MFpressure.define(stateout.boxArray(), stateout.DistributionMap(), 1, 2);
    MFb.define(stateout.boxArray(), stateout.DistributionMap(), 1, 0);
    MFscaledenth_cc.define(stateout.boxArray(), stateout.DistributionMap(), 1,
                           1);
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

    // First advance with Rusanov flux and compute scaled enthalpy
    // We do this on a box grown by 1 (but without overlapping between tiles)
    MFIter_loop(
        time, geom, statein, MFex, fluxes, dt,
        [&](const amrex::MFIter &mfi, const amrex::Real time,
            const amrex::Box & /*bx*/,
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
            ParallelFor(gbx,
                        [=] AMREX_GPU_DEVICE(int i, int j, int k)
                        {
                            scaledenth_cc(i, j, k)
                                = specific_enthalpy(i, j, k, consv_ex_arr)
                                  / consv_ex_arr(i, j, k, 0);

                            p(i, j, k) = pressure(i, j, k, consv_ex_arr);
                        });
        });

    // Fill boundary cells for pressure
    // TODO: check if this initialises pressure on course fine boundaries!
    FillDomainBoundary(
        MFpressure, geom,
        { AMREX_D_DECL(BCRec(BCType::foextrap, BCType::foextrap),
                       BCRec(BCType::foextrap, BCType::foextrap),
                       BCRec(BCType::foextrap, BCType::foextrap)) });

    BL_PROFILE_VAR_STOP(pexp);
    BL_PROFILE_VAR("advance_o1_pimex() assembly", passembly);

    // Assembly of source vector and setting coefficients

    const Real                  epsilon = AmrLevelAdv::h_prob_parm->epsilon;
    const Real                  adia    = AmrLevelAdv::h_prob_parm->adiabatic;
    GpuArray<Real, BL_SPACEDIM> dx      = geom.CellSizeArray();

    // Calculate source vector (cannot be done in MFIter loop above because it
    // requires offset values for the explicitly updated momentum densities)
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    {
        for (MFIter mfi(MFb, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box               &bx       = mfi.tilebox();
            const Array4<Real>       b        = MFb.array(mfi);
            const Array4<const Real> consv_ex = MFex.array(mfi);
            const Array4<const Real> sh_cc    = MFscaledenth_cc.array(mfi);

            Print() << std::endl << "b" << std::endl;
            ParallelFor(
                bx,
                [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    b(i, j, k)
                        = epsilon
                          * (consv_ex(i, j, k, 1 + AMREX_SPACEDIM)
                             - kinetic_energy(i, j, k, consv_ex)
                             - 0.5 * epsilon * dt
                                   * (AMREX_D_TERM(
                                       (sh_cc(i + 1, j, k)
                                            * consv_ex(i + 1, j, k, 1)
                                        - sh_cc(i - 1, j, k)
                                              * consv_ex(i - 1, j, k, 1))
                                           / dx[0],
                                       +(sh_cc(i, j + 1, k)
                                             * consv_ex(i, j + 1, k, 2)
                                         - sh_cc(i, j - 1, k)
                                               * consv_ex(i, j - 1, k, 2))
                                           / dx[1],
                                       +(sh_cc(i, j, k + 1)
                                             * consv_ex(i, j, k + 1, 3)
                                         - sh_cc(i, j, k - 1)
                                               * consv_ex(i, j, k - 1, 3))
                                           / dx[2])));
                });
        }
    }

    // An operator representing an equation of the form
    // (A * alpha(x) - B div (beta(x) grad)) * phi = f
    MLABecLaplacian pressure_op({ geom }, { stateout.boxArray() },
                                { stateout.DistributionMap() });

    // BCs
    // TODO: remove hard coding of this
    pressure_op.setDomainBC(
        { AMREX_D_DECL(LinOpBCType::Neumann, LinOpBCType::Neumann,
                       LinOpBCType::Neumann) },
        { AMREX_D_DECL(LinOpBCType::Neumann, LinOpBCType::Neumann,
                       LinOpBCType::Neumann) });

    if (level > 0)
        pressure_op.setCoarseFineBC(&crse_pressure, crse_ratio[0]);

    // pure homogeneous Neumann BC so we pass nullptr
    pressure_op.setLevelBC(0, nullptr);

    // set coefficients

    pressure_op.setScalars(1, dt * dt);
    pressure_op.setACoeffs(0, epsilon / (adia - 1));
    amrex::average_cellcenter_to_face(GetArrOfPtrs(MFscaledenth_f),
                                      MFscaledenth_cc, geom);
    pressure_op.setBCoeffs(0, amrex::GetArrOfConstPtrs(MFscaledenth_f));

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
    AMREX_ASSERT(!MFscaledenth_f[0].contains_nan());

    // TODO: use hypre here
    solver.solve({ &MFpressure }, { &MFb }, TOL_RES, TOL_ABS);

    // TODO: fill course fine boundary cells for pressure!
    FillDomainBoundary(
        MFpressure, geom,
        { AMREX_D_DECL(BCRec(BCType::foextrap, BCType::foextrap),
                       BCRec(BCType::foextrap, BCType::foextrap),
                       BCRec(BCType::foextrap, BCType::foextrap)) });

#ifdef DEBUG_PRINT
    Print() << std::endl << "exp quantities: " << std::endl;
    for (MFIter mfi(MFex, false); mfi.isValid(); ++mfi)
    {
        const Box bx = amrex::grow(mfi.validbox(), 1);
        for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i)
        {
            Print() << MFex.array(mfi)(i, 0, 0, 0) << "\t"
                    << MFex.array(mfi)(i, 0, 0, 1) << "\t"
                    << MFex.array(mfi)(i, 0, 0, 2) << std::endl;
        }
    }
    Print() << std::endl;

    Print() << std::endl << "Scaled enthalpy (cc): " << std::endl;
    for (MFIter mfi(MFscaledenth_cc, false); mfi.isValid(); ++mfi)
    {
        const Box bx = mfi.growntilebox();
        for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i)
        {
            Print() << MFscaledenth_cc.array(mfi)(i, 0, 0) << ", ";
        }
    }
    Print() << std::endl;

    Print() << std::endl << "Scaled enthalpy (f): " << std::endl;
    for (MFIter mfi(MFscaledenth_f[0], false); mfi.isValid(); ++mfi)
    {
        const Box bx = mfi.growntilebox();
        for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i)
        {
            Print() << MFscaledenth_f[0].array(mfi)(i, 0, 0) << ", ";
        }
    }
    Print() << std::endl;

    Print() << std::endl << "b: " << std::endl;
    for (MFIter mfi(MFb, false); mfi.isValid(); ++mfi)
    {
        const Box bx = mfi.growntilebox();
        for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i)
        {
            Print() << MFb.array(mfi)(i, 0, 0) << ", ";
        }
    }
    Print() << std::endl;

    Print() << std::endl << "Pressure: " << std::endl;
    for (MFIter mfi(MFpressure, false); mfi.isValid(); ++mfi)
    {
        const Box bx = mfi.growntilebox();
        for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i)
        {
            Print() << MFpressure.array(mfi)(i, 0, 0) << ", ";
        }
    }
    Print() << std::endl;
#endif

    BL_PROFILE_VAR_STOP(ppressure);
    BL_PROFILE_VAR("advance_o1_pimex() updating", pupdate);

    // TODO: USE AMREX BUILT-INS getGradSolution AND getFluxes

    // 1: Update the momentum densities in MFex to be the final momentum
    // densities, and update the scaled enthalpies
    // 2: Use the density and momentum densities in MFex to calculate the
    // final fluxes and final state

    // 1
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    {
        AMREX_D_TERM(const Real hdtdxe = dt / (2 * dx[0] * epsilon);
                     , const Real hdtdye = dt / (2 * dx[1] * epsilon);
                     , const Real hdtdze = dt / (2 * dx[2] * epsilon);)

        for (MFIter mfi(MFex, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box  &bx    = mfi.growntilebox();
            const auto &ex    = MFex.array(mfi);
            const auto &p     = MFpressure.const_array(mfi);
            const auto &sh_cc = MFscaledenth_cc.array(mfi);

            ParallelFor(
                bx,
                [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    AMREX_D_TERM(
                        ex(i, j, k, 1)
                        -= hdtdxe * (p(i + 1, j, k) - p(i - 1, j, k));
                        , ex(i, j, k, 2)
                          -= hdtdye * (p(i, j + 1, k) - p(i, j - 1, k));
                        , ex(i, j, k, 3)
                          -= hdtdze * (p(i, j, k + 1) - p(i, j, k - 1));)

                    sh_cc(i, j, k)
                        = (adia / (adia - 1)) * p(i, j, k) / ex(i, j, k, 0);
                });
        }
    }

    // 2
    MFIter_loop(
        time, geom, MFex, stateout, fluxes, dt,
        [&](const amrex::MFIter &mfi, const amrex::Real /*time*/,
            const amrex::Box    &bx,
            amrex::GpuArray<amrex::Box, AMREX_SPACEDIM> nbx,
            const amrex::FArrayBox &consv_ex, amrex::FArrayBox &stateout,
            AMREX_D_DECL(amrex::FArrayBox & fx, amrex::FArrayBox & fy,
                         amrex::FArrayBox & fz),
            amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx,
            const amrex::Real                            dt)
        {
            const Array4<const Real> ex    = consv_ex.array();
            const Array4<Real>       out   = stateout.array();
            const Array4<const Real> p     = MFpressure.array(mfi);
            const Array4<const Real> sh_cc = MFscaledenth_cc.array(mfi);
            AMREX_D_TERM(const Array4<Real> fluxx = fx.array();
                         , const Array4<Real> fluxy = fy.array();
                         , const Array4<Real> fluxz = fz.array();)

            const Real heps = epsilon / 2;
            AMREX_D_TERM(const Real hdtdx = 0.5 * dt / dx[0];
                         , const Real hdtdy = 0.5 * dt / dx[1];
                         , const Real hdtdz = 0.5 * dt / dx[2];)

            // Update
            ParallelFor(
                bx,
                [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    out(i, j, k, 0) = ex(i, j, k, 0);
                    AMREX_D_TERM(out(i, j, k, 1) = ex(i, j, k, 1);
                                 , out(i, j, k, 2) = ex(i, j, k, 2);
                                 , out(i, j, k, 3) = ex(i, j, k, 3);)
                    out(i, j, k, 1 + AMREX_SPACEDIM)
                        = ex(i, j, k, 1 + AMREX_SPACEDIM)
                          - (AMREX_D_TERM(
                              hdtdx
                                  * (sh_cc(i + 1, j, k) * ex(i + 1, j, k, 1)
                                     - sh_cc(i - 1, j, k)
                                           * ex(i - 1, j, k, 1)),
                              +hdtdy
                                  * (sh_cc(i, j + 1, k) * ex(i, j + 1, k, 2)
                                     - sh_cc(i, j - 1, k)
                                           * ex(i, j - 1, k, 2)),
                              +hdtdz
                                  * (sh_cc(i, j, k + 1) * ex(i, j, k + 1, 3)
                                     - sh_cc(i, j, k - 1)
                                           * ex(i, j, k - 1, 3))));
                });

            // Update fluxes
            ParallelFor(
                AMREX_D_DECL(nbx[0], nbx[1], nbx[2]),
                AMREX_D_DECL(
                    [=] AMREX_GPU_DEVICE(int i, int j, int k)
                    {
                        fluxx(i, j, k, 1)
                            += heps * (p(i, j, k) + p(i - 1, j, k));
                        fluxx(i, j, k, 1 + AMREX_SPACEDIM)
                            += 0.5
                               * (sh_cc(i, j, k) * ex(i, j, k, 1)
                                  + sh_cc(i - 1, j, k) * ex(i - 1, j, k, 1));
                    },
                    [=] AMREX_GPU_DEVICE(int i, int j, int k)
                    {
                        fluxy(i, j, k, 2)
                            += heps * (p(i, j, k) + p(i, j - 1, k));
                        fluxy(i, j, k, 1 + AMREX_SPACEDIM)
                            += 0.5
                               * (sh_cc(i, j, k) * ex(i, j, k, 2)
                                  + sh_cc(i, j - 1, k) * ex(i, j - 1, k, 2));
                    },
                    [=] AMREX_GPU_DEVICE(int i, int j, int k)
                    {
                        fluxz(i, j, k, 3)
                            += heps * (p(i, j, k) + p(i, j, k - 1));
                        fluxz(i, j, k, 1 + AMREX_SPACEDIM)
                            += 0.5
                               * (sh_cc(i, j, k) * ex(i, j, k, 3)
                                  + sh_cc(i, j, k - 1) * ex(i, j, k - 1, 3));
                    }));
        });

    BL_PROFILE_VAR_STOP(pupdate);
}

AMREX_GPU_HOST
void advance_rusanov_adv(const amrex::Real /*time*/, const amrex::Box &bx,
                         amrex::GpuArray<amrex::Box, AMREX_SPACEDIM> nbx,
                         const amrex::FArrayBox                     &statein,
                         amrex::FArrayBox                           &stateout,
                         AMREX_D_DECL(amrex::FArrayBox &fx,
                                      amrex::FArrayBox &fy,
                                      amrex::FArrayBox &fz),
                         amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx,
                         const amrex::Real                            dt)
{
    const int     NSTATE  = AmrLevelAdv::NUM_STATE;
    constexpr int STENSIL = 1;
    const Box     gbx     = amrex::grow(bx, STENSIL);
    // Temporary fab
    FArrayBox tmpfab;
    const int ntmpcomps = 2 * NSTATE * AMREX_SPACEDIM;
    // Using the Async Arena stops temporary fab from having its destructor
    // called too early. It is equivalent on old versions of CUDA to using
    // Elixir
    tmpfab.resize(gbx, ntmpcomps, The_Async_Arena());

    GpuArray<Array4<Real>, AMREX_SPACEDIM> adv_fluxes{ AMREX_D_DECL(
        tmpfab.array(0, NSTATE), tmpfab.array(NSTATE, NSTATE),
        tmpfab.array(NSTATE * 2, NSTATE)) };

    // Temporary fluxes. Need to be declared here instead of using passed
    // fluxes as otherwise we will get a race condition
    GpuArray<Array4<Real>, AMREX_SPACEDIM> tfluxes{ AMREX_D_DECL(
        tmpfab.array(NSTATE * 3, NSTATE), tmpfab.array(NSTATE * 4, NSTATE),
        tmpfab.array(NSTATE * 5, NSTATE)) };

    // Const version of above temporary fluxes
    GpuArray<Array4<const Real>, AMREX_SPACEDIM> tfluxes_c{ AMREX_D_DECL(
        tfluxes[0], tfluxes[1], tfluxes[2]) };

    Array4<const Real> consv_in  = statein.array();
    Array4<Real>       consv_out = stateout.array();

    GpuArray<Array4<Real>, AMREX_SPACEDIM> fluxes{ AMREX_D_DECL(
        fx.array(), fy.array(), fz.array()) };

    // Compute physical fluxes
    ParallelFor(gbx,
                [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    AMREX_D_TERM(
                        adv_flux<0>(i, j, k, consv_in, adv_fluxes[0]);
                        , adv_flux<1>(i, j, k, consv_in, adv_fluxes[1]);
                        , adv_flux<2>(i, j, k, consv_in, adv_fluxes[2]);)
                });

    // Compute Rusanov fluxes
    for (int d = 0; d < AMREX_SPACEDIM; d++)
    {
        const int i_off = (d == 0) ? 1 : 0;
        const int j_off = (d == 1) ? 1 : 0;
        const int k_off = (d == 2) ? 1 : 0;
        ParallelFor(
            amrex::surroundingNodes(bx, d), NSTATE,
            [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
            {
                const Real u_max = amrex::max(
                    consv_in(i, j, k, 1 + d) / consv_in(i, j, k, 0),
                    consv_in(i - i_off, j - j_off, k - k_off, 1 + d)
                        / consv_in(i - i_off, j - j_off, k - k_off, 0));
                tfluxes[d](i, j, k, n)
                    = 0.5
                      * ((adv_fluxes[d](i, j, k, n)
                          + adv_fluxes[d](i - i_off, j - j_off, k - k_off, n))
                         - u_max
                               * (consv_in(i, j, k, n)
                                  - consv_in(i - i_off, j - j_off, k - k_off,
                                             n)));
            });
    }

    // Conservative update
    AMREX_D_TERM(const Real dtdx = dt / dx[0];, const Real dtdy = dt / dx[1];
                 , const Real                              dtdz = dt / dx[2];)
    ParallelFor(bx, NSTATE,
                [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                {
                    consv_out(i, j, k, n)
                        = consv_in(i, j, k, n)
                          - AMREX_D_TERM(dtdx
                                             * (tfluxes[0](i + 1, j, k, n)
                                                - tfluxes[0](i, j, k, n)),
                                         +dtdy
                                             * (tfluxes[1](i, j + 1, k, n)
                                                - tfluxes[1](i, j, k, n)),
                                         +dtdz
                                             * (tfluxes[2](i, j, k + 1, n)
                                                - tfluxes[2](i, j, k, n)));
                });

    // Copy temporary fluxes onto potentially smaller nodal tile
    for (int d = 0; d < AMREX_SPACEDIM; d++)
        ParallelFor(nbx[d], NSTATE,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                    { fluxes[d](i, j, k, n) = tfluxes_c[d](i, j, k, n); });
}

AMREX_GPU_HOST
void copy_pressure(const amrex::MultiFab &statesrc, amrex::MultiFab &dst)
{
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