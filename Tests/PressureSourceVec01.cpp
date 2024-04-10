#include "BoxTest.H"
#include "IMEX/IMEX_O1.H"
#include "IMEX_K/euler_K.H"
#include <gtest/gtest.h>

using namespace amrex;

#define ASSERT_IF_CONTAINS(arr, i, j, k, val, TOL)                            \
    if (arr.contains(i, j, k))                                                \
    {                                                                         \
        EXPECT_NEAR(arr(i, j, k), val, TOL);                                  \
    }

#if AMREX_SPACEDIM == 3
TEST_F(BoxTest, PressureSourceVector)
{
    setup(true);
    const Real epsilon = AmrLevelAdv::h_prob_parm->epsilon;
    const Real adia    = AmrLevelAdv::h_prob_parm->adiabatic;
    ASSERT_EQ(epsilon, 0.5);
    ASSERT_EQ(adia, 1.6);

    const Real dt     = 0.1;
    const Real hedtdx = 0.5 * epsilon * dt / dx[0];
    const Real hedtdy = 0.5 * epsilon * dt / dx[1];
    const Real hedtdz = 0.5 * epsilon * dt / dx[2];

    MultiFab MFscaledenth_cc(ba, dm, 1, 1);
    MultiFab MFb(ba, dm, 1, 0);

    for (MFIter mfi(MFscaledenth_cc, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box   gbx           = mfi.growntilebox();
        const auto &consv         = statein.const_array(mfi);
        const auto &scaledenth_cc = MFscaledenth_cc.array(mfi);
        ParallelFor(gbx,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k)
                    {
                        scaledenth_cc(i, j, k)
                            = specific_enthalpy(i, j, k, consv)
                              / consv(i, j, k, 0);
                        if (i < 2 && j < 2 && k < 2)
                            EXPECT_NEAR(scaledenth_cc(i, j, k), 0.8, 1e-12);
                        else
                            EXPECT_NEAR(scaledenth_cc(i, j, k), 3.2, 1e-12);
                    });
    }

    pressure_source_vector(statein, MFscaledenth_cc, MFb, hedtdx, hedtdy,
                           hedtdz);

    for (MFIter mfi(MFscaledenth_cc, false); mfi.isValid(); ++mfi)
    {
        const auto &b = MFb.const_array(mfi);
        // along edges
        ASSERT_IF_CONTAINS(b, 0, 0, 0, 0.5, 1e-12);
        ASSERT_IF_CONTAINS(b, 1, 0, 0, 0.5 + dt / dx[0], 1e-12);
        ASSERT_IF_CONTAINS(b, 2, 0, 0, 1 + dt / dx[0], 1e-12);
        ASSERT_IF_CONTAINS(b, 3, 0, 0, 1, 1e-12);
        ASSERT_IF_CONTAINS(b, 0, 1, 0, 0.5, 1e-12);
        ASSERT_IF_CONTAINS(b, 0, 2, 0, 1, 1e-12);
        ASSERT_IF_CONTAINS(b, 0, 3, 0, 1, 1e-12);
        ASSERT_IF_CONTAINS(b, 0, 0, 1, 0.5 - dt / dx[2], 1e-12);
        ASSERT_IF_CONTAINS(b, 0, 0, 2, 1 - dt / dx[2], 1e-12);
        ASSERT_IF_CONTAINS(b, 0, 0, 3, 1, 1e-12);

        // boring terms
        ASSERT_IF_CONTAINS(b, 1, 2, 0, 1, 1e-12);
        ASSERT_IF_CONTAINS(b, 1, 2, 1, 1, 1e-12);
        ASSERT_IF_CONTAINS(b, 1, 3, 0, 1, 1e-12);

        // less boring terms
        ASSERT_IF_CONTAINS(b, 1, 1, 0, 0.5 + dt / dx[0], 1e-12);
        ASSERT_IF_CONTAINS(b, 1, 1, 1, 0.5 + dt / dx[0] - dt / dx[2], 1e-12);
        ASSERT_IF_CONTAINS(b, 1, 0, 1, 0.5 + dt / dx[0] - dt / dx[2], 1e-12);
        ASSERT_IF_CONTAINS(b, 0, 1, 1, 0.5 - dt / dx[2], 1e-12);
        ASSERT_IF_CONTAINS(b, 2, 1, 1, 1 + dt / dx[0], 1e-12);
        ASSERT_IF_CONTAINS(b, 1, 2, 1, 1, 1e-12);
        ASSERT_IF_CONTAINS(b, 1, 1, 2, 1 - dt / dx[2], 1e-12);
    }
}
#endif