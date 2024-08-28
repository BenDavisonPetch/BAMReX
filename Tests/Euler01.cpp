#include <AMReX_AmrLevel.H>
#include <gtest/gtest.h>

#include "BoxTest.H"

using namespace amrex;

#if AMREX_SPACEDIM == 2
TEST_F(BoxTest, ConsvFromPrimvArray2D)
#elif AMREX_SPACEDIM == 3
TEST_F(BoxTest, ConsvFromPrimvArray3D)
#endif
{
    setup(false);
    const Real epsilon      = AmrLevelAdv::h_parm->epsilon;
    const Real adiabatic    = AmrLevelAdv::h_parm->adiabatic;
    const auto consv_L_calc = consv_from_primv(primv_L, adiabatic, epsilon);
    const auto consv_R_calc = consv_from_primv(primv_R, adiabatic, epsilon);
    for (int n = 0; n < EULER_NCOMP; ++n)
    {
        EXPECT_NEAR(consv_L[n], consv_L_calc[n], 1e-12);
        EXPECT_NEAR(consv_R[n], consv_R_calc[n], 1e-12);
    }
}

#if AMREX_SPACEDIM == 2
TEST_F(BoxTest, PrimvFromConsv2D)
#elif AMREX_SPACEDIM == 3
TEST_F(BoxTest, PrimvFromConsv3D)
#endif
{
    setup(false);
    const Real epsilon   = AmrLevelAdv::h_parm->epsilon;
    const Real adiabatic = AmrLevelAdv::h_parm->adiabatic;
    const auto primv_L_calc
        = primv_from_consv(statein[0].array(), adiabatic, epsilon, 0, 0, 0);
    const auto primv_R_calc
        = primv_from_consv(statein[0].array(), adiabatic, epsilon, 3, 0, 0);
    for (int n = 0; n < EULER_NCOMP; ++n)
    {
        EXPECT_NEAR(primv_L[n], primv_L_calc[n], 1e-12);
        EXPECT_NEAR(primv_R[n], primv_R_calc[n], 1e-12);
    }
}

#if AMREX_SPACEDIM >= 2
#if AMREX_SPACEDIM == 2
TEST_F(BoxTest, MaxWaveSpeed2D)
#elif AMREX_SPACEDIM == 3
TEST_F(BoxTest, MaxWaveSpeed3D)
#endif
{
    setup(true);
    ASSERT_EQ(AmrLevelAdv::h_parm->epsilon, 0.5);
    ASSERT_EQ(AmrLevelAdv::h_parm->adiabatic, 1.6);
    const Real speed = max_wave_speed(0, statein);
#if AMREX_SPACEDIM == 3
    // |v|_L = sqrt(14)
    // c_L = sqrt(0.48)
    // |v|_R = sqrt(21)
    // c_R = sqrt(1.92)
    const Real expected = sqrt(21) + sqrt(1.92 / 0.5);
#elif AMREX_SPACEDIM == 2
    // |v|_L = sqrt(5)
    // c_L = sqrt(2.64)
    // |v|_R = sqrt(5)
    // c_R = sqrt(5.76)
    const Real expected = sqrt(5) + sqrt(5.76 / 0.5);
#endif
    ASSERT_TRUE(speed > 0);
    ASSERT_TRUE(expected > 0);
    EXPECT_NEAR(speed, expected, 1e-12 * expected);
}
#endif