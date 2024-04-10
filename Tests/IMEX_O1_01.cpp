#include "BoxTest.H"
#include "IMEX/IMEX_O1.H"
#include "IMEX_K/euler_K.H"
#include <AMReX_DistributionMapping.H>
#include <gtest/gtest.h>

using namespace amrex;

TEST_F(BoxTest, IMEX_O1_FluxMatching)
{
    setup(true);
    const Real epsilon = AmrLevelAdv::h_prob_parm->epsilon;
    const Real adia    = AmrLevelAdv::h_prob_parm->adiabatic;
    ASSERT_EQ(epsilon, 0.5);
    ASSERT_EQ(adia, 1.6);

    ASSERT_FALSE(statein.contains_nan());

    const Real dt = 0.01;
    IntVect    crse_ratio{ AMREX_D_DECL(1, 1, 1) };
    advance_o1_pimex(0, crse_ratio, 0, geom, statein, stateout, fluxes,
                     nullptr, dt, bcs, 1, 0);

    ASSERT_FALSE(stateout.contains_nan());
    ASSERT_FALSE(fluxes[0].contains_nan());
    ASSERT_FALSE(fluxes[1].contains_nan());
#if AMREX_SPACEDIM == 3
    ASSERT_FALSE(fluxes[2].contains_nan());
#endif

    for (MFIter mfi(stateout, false); mfi.isValid(); ++mfi)
    {
        const Box  &bx    = mfi.validbox();
        const auto &out   = stateout.const_array(mfi);
        const auto &in    = statein.const_array(mfi);
        const auto &fluxx = fluxes[0].const_array(mfi);
        const auto &fluxy = fluxes[1].const_array(mfi);
        const Real  dtdx  = dt / dx[0];
        const Real  dtdy  = dt / dx[1];
#if AMREX_SPACEDIM == 3
        const auto &fluxz = fluxes[2].const_array(mfi);
        const Real  dtdz  = dt / dx[2];
#endif
        ParallelFor(bx,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k)
                    {
                        // Check output corresponds to flux
                        // Print() << "i j k = " << i << " " << j << " " << k
                        // << " "
                        //         << std::endl;
                        for (int n = 0; n < EULER_NCOMP; ++n)
                        {
                            // Print() << "\tn = " << n << std::endl;
                            EXPECT_NEAR(out(i, j, k, n),
                                        in(i, j, k, n)
                                            - (AMREX_D_TERM(
                                                dtdx
                                                    * (fluxx(i + 1, j, k, n)
                                                       - fluxx(i, j, k, n)),
                                                +dtdy
                                                    * (fluxy(i, j + 1, k, n)
                                                       - fluxy(i, j, k, n)),
                                                +dtdz
                                                    * (fluxz(i, j, k + 1, n)
                                                       - fluxz(i, j, k, n)))),
                                        fabs(out(i, j, k, n)) * 1e-12);
                        }
                    });
    }
}

TEST_F(BoxTest, IMEX_O1_GridSplitting)
{
    const Real dt = 0.01;
    IntVect    crse_ratio{ AMREX_D_DECL(1, 1, 1) };

    // compute first without grid splitting
    setup(false);
    advance_o1_pimex(0, crse_ratio, 0, geom, statein, stateout, fluxes,
                     nullptr, dt, bcs, 1, 0);

    BoxArray            ba1 = stateout.boxArray();
    DistributionMapping dm1 = stateout.DistributionMap();
    MultiFab            MFsoln1(ba1, dm1, EULER_NCOMP, 0);
    MultiFab::Copy(MFsoln1, stateout, 0, 0, EULER_NCOMP, 0);
    TearDown();
    // compute with grid splitting
    setup(true);
    advance_o1_pimex(0, crse_ratio, 0, geom, statein, stateout, fluxes,
                     nullptr, dt, bcs, 1, 0);

    // compare solutions
    MFIter::allowMultipleMFIters(true);
    for (MFIter mfi(stateout, false); mfi.isValid(); ++mfi)
    {
        MFIter      mfi1(MFsoln1, false);
        const Box  &bx    = mfi.validbox();
        const auto &soln1 = MFsoln1.const_array(mfi1);
        const auto &soln2 = stateout.const_array(mfi);

        For(bx, EULER_NCOMP,
            [=](int i, int j, int k, int n)
            {
                EXPECT_NEAR(soln1(i, j, k, n), soln2(i, j, k, n),
                            fabs(soln1(i, j, k, n) * 1e-12));
            });
    }
}