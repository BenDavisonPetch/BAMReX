

#include <gtest/gtest.h>

#include "Fluxes/HLLC.H"
#include "hllc_test.H"

TEST_F(Toro2020Test1, WaveSpeedEstToro1994)
{
    Real S_L, S_interm, S_R;
    estimate_wave_speed_HLLC_Toro<0>(1, 0, 0, dataL, dataR, adiabatic,
                                     adiabatic, 1, S_L, S_R, S_interm);
    EXPECT_NEAR(S_L, exp_To_S_L, 0.0001);
    EXPECT_NEAR(S_R, exp_To_S_R, 0.0001);
    EXPECT_NEAR(S_interm, exp_To_S_interm, 0.0001);
}

TEST_F(Toro2020Test2, WaveSpeedEstToro1994)
{
    Real S_L, S_interm, S_R;
    estimate_wave_speed_HLLC_Toro<0>(1, 0, 0, dataL, dataR, adiabatic,
                                     adiabatic, 1, S_L, S_R, S_interm);
    EXPECT_NEAR(S_L, exp_To_S_L, 0.0001);
    EXPECT_NEAR(S_R, exp_To_S_R, 0.0001);
    EXPECT_NEAR(S_interm, exp_To_S_interm, 0.0001);
}

TEST_F(Toro2020Test4, WaveSpeedEstToro1994)
{
    Real S_L, S_interm, S_R;
    estimate_wave_speed_HLLC_Toro<0>(1, 0, 0, dataL, dataR, adiabatic,
                                     adiabatic, 1, S_L, S_R, S_interm);
    EXPECT_NEAR(S_L, exp_To_S_L, 0.0001); // TODO: check if this is reasonable
    EXPECT_NEAR(S_R, exp_To_S_R, 0.0001);
    EXPECT_NEAR(S_interm, exp_To_S_interm, 0.0005);
}

TEST_F(Toro2020Test5, WaveSpeedEstToro1994)
{
    Real S_L, S_interm, S_R;
    estimate_wave_speed_HLLC_Toro<0>(1, 0, 0, dataL, dataR, adiabatic,
                                     adiabatic, 1, S_L, S_R, S_interm);
    EXPECT_NEAR(S_L, exp_To_S_L, 0.0001);
    EXPECT_NEAR(S_R, exp_To_S_R, 0.0001);
    EXPECT_NEAR(S_interm, exp_To_S_interm, 0.002);
}

TEST_F(Toro2020Test7, WaveSpeedEstToro1994)
{
    Real S_L, S_interm, S_R;
    estimate_wave_speed_HLLC_Toro<0>(1, 0, 0, dataL, dataR, adiabatic,
                                     adiabatic, 1, S_L, S_R, S_interm);
    EXPECT_NEAR(S_L, exp_To_S_L, 0.0001);
    EXPECT_NEAR(S_R, exp_To_S_R, 0.0001);
    EXPECT_NEAR(S_interm, exp_To_S_interm, 0.0001);
}

TEST_F(Toro2020Test5Scaled, WaveSpeedEstToro1994)
{
    Real S_L, S_interm, S_R;
    estimate_wave_speed_HLLC_Toro<0>(1, 0, 0, dataL, dataR, adiabatic,
                                     adiabatic, epsilon, S_L, S_R, S_interm);
    EXPECT_NEAR(S_L, exp_To_S_L / u_0, 0.0001 / u_0);
    EXPECT_NEAR(S_R, exp_To_S_R / u_0, 0.0001 / u_0);
    EXPECT_NEAR(S_interm, exp_To_S_interm / u_0, 0.002 / u_0);
}

TEST_F(Toro2020Test1, WaveSpeedEstTMSb)
{
    Real S_L, S_interm, S_R;
    estimate_wave_speed_HLLC_TMSb<0>(1, 0, 0, dataL, dataR, adiabatic,
                                     adiabatic, 1, S_L, S_R, S_interm);
    EXPECT_NEAR(S_L, exp_TMSb_S_L, 0.0001);
    EXPECT_NEAR(S_R, exp_TMSb_S_R, 0.0001);
    EXPECT_NEAR(S_interm, exp_TMSb_S_interm, 0.0001);
}

TEST_F(Toro2020Test2, WaveSpeedEstTMSb)
{
    Real S_L, S_interm, S_R;
    estimate_wave_speed_HLLC_TMSb<0>(1, 0, 0, dataL, dataR, adiabatic,
                                     adiabatic, 1, S_L, S_R, S_interm);
    EXPECT_NEAR(S_L, exp_TMSb_S_L, 0.0001);
    EXPECT_NEAR(S_R, exp_TMSb_S_R, 0.0001);
    EXPECT_NEAR(S_interm, exp_TMSb_S_interm, 0.0001);
}

TEST_F(Toro2020Test4, WaveSpeedEstTMSb)
{
    Real S_L, S_interm, S_R;
    estimate_wave_speed_HLLC_TMSb<0>(1, 0, 0, dataL, dataR, adiabatic,
                                     adiabatic, 1, S_L, S_R, S_interm);
    EXPECT_NEAR(S_L, exp_TMSb_S_L, 2.5); // TODO: check if this is reasonable
    EXPECT_NEAR(S_R, exp_TMSb_S_R, 0.0001);
    EXPECT_NEAR(S_interm, exp_TMSb_S_interm, 0.6);
}

TEST_F(Toro2020Test5, WaveSpeedEstTMSb)
{
    Real S_L, S_interm, S_R;
    estimate_wave_speed_HLLC_TMSb<0>(1, 0, 0, dataL, dataR, adiabatic,
                                     adiabatic, 1, S_L, S_R, S_interm);
    EXPECT_NEAR(S_L, exp_TMSb_S_L, 0.008);
    EXPECT_NEAR(S_R, exp_TMSb_S_R, 0.0085);
    EXPECT_NEAR(S_interm, exp_TMSb_S_interm, 0.002);
}

TEST_F(Toro2020Test7, WaveSpeedEstTMSb)
{
    Real S_L, S_interm, S_R;
    estimate_wave_speed_HLLC_TMSb<0>(1, 0, 0, dataL, dataR, adiabatic,
                                     adiabatic, 1, S_L, S_R, S_interm);
    EXPECT_NEAR(S_L, exp_TMSb_S_L, 0.0001);
    EXPECT_NEAR(S_R, exp_TMSb_S_R, 0.0001);
    EXPECT_NEAR(S_interm, exp_TMSb_S_interm, 0.0001);
}

TEST_F(Toro2020Test5Scaled, WaveSpeedEstTMSb)
{
    Real S_L, S_interm, S_R;
    estimate_wave_speed_HLLC_TMSb<0>(1, 0, 0, dataL, dataR, adiabatic,
                                     adiabatic, epsilon, S_L, S_R, S_interm);
    EXPECT_NEAR(S_L, exp_TMSb_S_L / u_0, 0.008 / u_0);
    EXPECT_NEAR(S_R, exp_TMSb_S_R / u_0, 0.0085 / u_0);
    EXPECT_NEAR(S_interm, exp_TMSb_S_interm / u_0, 0.002 / u_0);
}