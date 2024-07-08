#include <AMReX_DistributionMapping.H>
#include <AMReX_MFIter.H>
#include <gtest/gtest.h>

#include "BoxTest.H"
#include "Fluxes/Update.H"
#include "IMEX/IMEXSettings.H"
#include "IMEX/IMEX_Fluxes.H"
#include "IMEX/IMEX_O1.H"
#include "IMEX_K/euler_K.H"

using namespace amrex;

class IMEXO1 : public BoxTest
{
  protected:
    MultiFab m_pressure;

    IMEXO1()
        : BoxTest()
    {
    }

    void setup(bool split_grid)
    {
        BoxTest::setup(split_grid);
        m_pressure.define(ba, dm, 1, 2);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        {
            for (MFIter mfi(statein, TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                const Box  &bx = mfi.growntilebox();
                const auto &p  = m_pressure.array(mfi);
                ParallelFor(bx,
                            [=] AMREX_GPU_DEVICE(int i, int j, int k)
                            {
                                if (i < 2 && j < 2 && k < 2)
                                    p(i, j, k) = primv_L[1 + AMREX_SPACEDIM];
                                else
                                    p(i, j, k) = primv_R[1 + AMREX_SPACEDIM];
                            });
            }
        }
    }
};

TEST_F(IMEXO1, SplitKEEnergyMatch)
{
    setup(true);
    const Real eps  = AmrLevelAdv::h_prob_parm->epsilon;
    const Real adia = AmrLevelAdv::h_prob_parm->adiabatic;
    ASSERT_EQ(eps, 0.5);
    ASSERT_EQ(adia, 1.6);

    ASSERT_FALSE(statein.contains_nan());

    const Real dt = 0.01;

    IMEXSettings settings;
    settings.linearise             = true;
    settings.ke_method             = IMEXSettings::split;
    settings.enthalpy_method       = IMEXSettings::reuse_pressure;
    settings.verbose               = true;
    settings.bottom_solver_verbose = true;
    settings.advection_flux        = IMEXSettings::rusanov;

    Print() << std::endl
            << "Testing for enthalpy_method = reuse_pressure..." << std::endl
            << std::endl;

    compute_flux_imex_o1(0, geom, statein, statein, m_pressure, fluxes, dt,
                         bcs, settings);

    ASSERT_FALSE(fluxes[0].contains_nan());
    ASSERT_FALSE(fluxes[1].contains_nan());
#if AMREX_SPACEDIM == 3
    ASSERT_FALSE(fluxes[2].contains_nan());
#endif

    conservative_update(geom, statein, fluxes, stateout, dt);

    for (MFIter mfi(stateout, false); mfi.isValid(); ++mfi)
    {
        const Box  &bx  = mfi.validbox();
        const auto &out = stateout.const_array(mfi);
        const auto &in  = statein.const_array(mfi);
        const auto &p   = m_pressure.const_array(mfi);
        ParallelFor(bx,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k)
                    {
                        Print() << i << " " << j << " " << k << std::endl;
                        const Real expected
                            = p(i, j, k) / (adia - 1)
                              + 0.5 * eps / in(i, j, k, 0)
                                    * (AMREX_D_TERM(
                                        in(i, j, k, 1) * out(i, j, k, 1),
                                        +in(i, j, k, 2) * out(i, j, k, 2),
                                        +in(i, j, k, 3) * out(i, j, k, 3)));

                        EXPECT_NEAR(expected, out(i, j, k, 1 + AMREX_SPACEDIM),
                                    fabs(expected) * 1e-12);
                    });
    }

    //     Print() << std::endl
    //             << "Testing for enthalpy_method = conservative..." <<
    //             std::endl
    //             << std::endl;

    //     settings.enthalpy_method = IMEXSettings::conservative;
    //     compute_flux_imex_o1(0, geom, statein, statein, m_pressure, fluxes,
    //     dt, bcs,
    //                          settings);

    //     ASSERT_FALSE(fluxes[0].contains_nan());
    //     ASSERT_FALSE(fluxes[1].contains_nan());
    // #if AMREX_SPACEDIM == 3
    //     ASSERT_FALSE(fluxes[2].contains_nan());
    // #endif

    //     conservative_update(geom, statein, fluxes, stateout, dt);

    //     for (MFIter mfi(stateout, false); mfi.isValid(); ++mfi)
    //     {
    //         const Box  &bx  = mfi.validbox();
    //         const auto &out = stateout.const_array(mfi);
    //         const auto &in  = statein.const_array(mfi);
    //         const auto &p   = m_pressure.const_array(mfi);
    //         ParallelFor(bx,
    //                     [=] AMREX_GPU_DEVICE(int i, int j, int k)
    //                     {
    //                         const Real expected
    //                             = p(i, j, k) / (adia - 1)
    //                               + 0.5 * eps / in(i, j, k, 0)
    //                                     * (AMREX_D_TERM(
    //                                         in(i, j, k, 1) * out(i, j, k,
    //                                         1), +in(i, j, k, 2) * out(i, j,
    //                                         k, 2), +in(i, j, k, 3) * out(i,
    //                                         j, k, 3)));

    //                         EXPECT_NEAR(expected, out(i, j, k, 1 +
    //                         AMREX_SPACEDIM),
    //                                     fabs(expected) * 1e-12);
    //                     });
    //     }

    //     Print() << std::endl
    //             << "Testing for enthalpy_method = conservative_ex..." <<
    //             std::endl
    //             << std::endl;

    //     settings.enthalpy_method = IMEXSettings::conservative_ex;
    //     compute_flux_imex_o1(0, geom, statein, statein, m_pressure, fluxes,
    //     dt, bcs,
    //                          settings);

    //     ASSERT_FALSE(fluxes[0].contains_nan());
    //     ASSERT_FALSE(fluxes[1].contains_nan());
    // #if AMREX_SPACEDIM == 3
    //     ASSERT_FALSE(fluxes[2].contains_nan());
    // #endif

    //     conservative_update(geom, statein, fluxes, stateout, dt);

    //     for (MFIter mfi(stateout, false); mfi.isValid(); ++mfi)
    //     {
    //         const Box  &bx  = mfi.validbox();
    //         const auto &out = stateout.const_array(mfi);
    //         const auto &in  = statein.const_array(mfi);
    //         const auto &p   = m_pressure.const_array(mfi);
    //         ParallelFor(bx,
    //                     [=] AMREX_GPU_DEVICE(int i, int j, int k)
    //                     {
    //                         const Real expected
    //                             = p(i, j, k) / (adia - 1)
    //                               + 0.5 * eps / in(i, j, k, 0)
    //                                     * (AMREX_D_TERM(
    //                                         in(i, j, k, 1) * out(i, j, k,
    //                                         1), +in(i, j, k, 2) * out(i, j,
    //                                         k, 2), +in(i, j, k, 3) * out(i,
    //                                         j, k, 3)));

    //                         EXPECT_NEAR(expected, out(i, j, k, 1 +
    //                         AMREX_SPACEDIM),
    //                                     fabs(expected) * 1e-12);
    //                     });
    //     }
}

TEST_F(IMEXO1, ExplicitKEEnergyMatch)
{
    setup(true);
    const Real eps  = AmrLevelAdv::h_prob_parm->epsilon;
    const Real adia = AmrLevelAdv::h_prob_parm->adiabatic;
    ASSERT_EQ(eps, 0.5);
    ASSERT_EQ(adia, 1.6);

    ASSERT_FALSE(statein.contains_nan());

    const Real dt = 0.01;

    IMEXSettings settings;
    settings.linearise             = true;
    settings.ke_method             = IMEXSettings::ex;
    settings.enthalpy_method       = IMEXSettings::reuse_pressure;
    settings.verbose               = true;
    settings.bottom_solver_verbose = true;
    settings.advection_flux        = IMEXSettings::rusanov;

    Print() << std::endl
            << "Testing for enthalpy_method = reuse_pressure..." << std::endl
            << std::endl;

    compute_flux_imex_o1(0, geom, statein, statein, m_pressure, fluxes, dt,
                         bcs, settings);

    ASSERT_FALSE(fluxes[0].contains_nan());
    ASSERT_FALSE(fluxes[1].contains_nan());
#if AMREX_SPACEDIM == 3
    ASSERT_FALSE(fluxes[2].contains_nan());
#endif

    conservative_update(geom, statein, fluxes, stateout, dt);

    for (MFIter mfi(stateout, false); mfi.isValid(); ++mfi)
    {
        const Box  &bx  = mfi.validbox();
        const auto &out = stateout.const_array(mfi);
        const auto &p   = m_pressure.const_array(mfi);

        FArrayBox                     consv_ex(bx, EULER_NCOMP);
        GpuArray<Box, AMREX_SPACEDIM> dummy;
        FArrayBox                     dummyfab;
        advance_rusanov_adv(0, bx, dummy, statein[mfi], statein[mfi], consv_ex,
                            AMREX_D_DECL(dummyfab, dummyfab, dummyfab), dx,
                            dt);
        const auto &ex = consv_ex.const_array();

        ParallelFor(
            bx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                Print() << i << " " << j << " " << k << std::endl;
                const Real expected
                    = p(i, j, k) / (adia - 1)
                      + 0.5 * eps / ex(i, j, k, 0)
                            * (AMREX_D_TERM(ex(i, j, k, 1) * ex(i, j, k, 1),
                                            +ex(i, j, k, 2) * ex(i, j, k, 2),
                                            +ex(i, j, k, 3) * ex(i, j, k, 3)));

                EXPECT_NEAR(expected, out(i, j, k, 1 + AMREX_SPACEDIM),
                            fabs(expected) * 1e-12);
            });
    }
}

TEST_F(IMEXO1, GridSplitting)
{
    const Real dt = 0.01;

    IMEXSettings settings;
    settings.linearise       = true;
    settings.ke_method       = IMEXSettings::ex;
    settings.enthalpy_method = IMEXSettings::conservative_ex;
    settings.advection_flux  = IMEXSettings::rusanov;

    // compute first without grid splitting
    setup(false);
    compute_flux_imex_o1(0, geom, statein, statein, m_pressure, fluxes, dt,
                         bcs, settings);

    Array<MultiFab, AMREX_SPACEDIM> nonsplit_fluxes;
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        nonsplit_fluxes[d].define(fluxes[d].boxArray(),
                                  fluxes[d].DistributionMap(), EULER_NCOMP, 0);
        MultiFab::Copy(nonsplit_fluxes[d], fluxes[d], 0, 0, EULER_NCOMP, 0);
    }

    TearDown();
    // compute with grid splitting
    setup(true);
    compute_flux_imex_o1(0, geom, statein, statein, m_pressure, fluxes, dt,
                         bcs, settings);

    // compare solutions
    MFIter::allowMultipleMFIters(true);
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        for (MFIter mfi(fluxes[d], false); mfi.isValid(); ++mfi)
        {
            MFIter      mfi1(nonsplit_fluxes[d], false);
            const Box  &bx    = mfi.validbox();
            const auto &soln1 = nonsplit_fluxes[d].const_array(mfi1);
            const auto &soln2 = fluxes[d].const_array(mfi);

            For(bx, EULER_NCOMP,
                [=](int i, int j, int k, int n)
                {
                    EXPECT_NEAR(soln1(i, j, k, n), soln2(i, j, k, n),
                                fabs(soln1(i, j, k, n) * 1e-12));
                });
        }
    }
}