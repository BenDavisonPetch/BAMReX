#include "BoxTest.H"
#include "IMEX/IMEX_O1.H"
#include "IMEX_K/euler_K.H"
#include <gtest/gtest.h>

using namespace amrex;

/**
 * Tests the update to the conservative variables after calculating pressure
 */
#if AMREX_SPACEDIM == 3
TEST_F(BoxTest, IMEX_O1_FluxMatching)
{
    const Real epsilon = AmrLevelAdv::h_prob_parm->epsilon;
    const Real adia    = AmrLevelAdv::h_prob_parm->adiabatic;
    ASSERT_EQ(epsilon, 0.5);
    ASSERT_EQ(adia, 1.6);

    const Real dt = 0.01;
    IntVect crse_ratio{1,1,1};
    advance_o1_pimex(0, crse_ratio, 0, geom, statein, stateout, fluxes, nullptr, dt, 1, 0);

    ASSERT_FALSE(stateout.contains_nan());
    ASSERT_FALSE(fluxes[0].contains_nan());
    ASSERT_FALSE(fluxes[1].contains_nan());
    ASSERT_FALSE(fluxes[2].contains_nan());

    for (MFIter mfi(stateout, false); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.validbox();
        const auto& out = stateout.const_array(mfi);
        const auto& in = statein.const_array(mfi);
        const auto& fluxx = fluxes[0].const_array(mfi);
        const auto& fluxy = fluxes[1].const_array(mfi);
        const auto& fluxz = fluxes[2].const_array(mfi);
        const Real dtdx = dt/dx[0];
        const Real dtdy = dt/dx[1];
        const Real dtdz = dt/dx[2];
        ParallelFor(
                bx,
                [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    // Check output corresponds to flux
                    // Print() << "i j k = " << i << " " << j << " " << k << " "
                    //         << std::endl;
                    for (int n = 0; n < EULER_NCOMP; ++n)
                    {
                        // Print() << "\tn = " << n << std::endl;
                        EXPECT_NEAR(
                            out(i, j, k, n),
                            in(i, j, k, n)
                                - (AMREX_D_TERM(dtdx
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
#endif