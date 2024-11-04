/**
 * Tests MHD functionality of HLLC
 */
#include "Fluxes/Fluxes.H"
#include "System/Plasma/Plasma_K.H"
#include "UnigridTest.H"

class HLLC_MHDTest : public UnigridTest
{
  protected:
    MultiFab     MF_U_L, MF_U_R;
    Array4<Real> U_L, U_R;

    const int i = 3;
    const int j = (AMREX_SPACEDIM > 1) ? 3 : 0;
    const int k = (AMREX_SPACEDIM > 2) ? 4 : 0;

    HLLC_MHDTest()
    {
        ParmParse pp("prob");
        pp.add("system", (std::string) "plasma");
        pp.add("adiabatic", 1.6);
        pp.add("epsilon", 1);
        IntVect                        dom_lo(AMREX_D_DECL(2, 3, 4));
        IntVect                        dom_hi(AMREX_D_DECL(4, 3, 4));
        GpuArray<Real, AMREX_SPACEDIM> prob_lo{ AMREX_D_DECL(0., 0., 0.) };
        GpuArray<Real, AMREX_SPACEDIM> prob_hi{ AMREX_D_DECL(0.2, 1., 1.) };
        setup(dom_lo, dom_hi, prob_lo, prob_hi, pp, pp);

        MF_U_L.define(ba, dm, PUInd::size, 0);
        MF_U_R.define(ba, dm, PUInd::size, 0);

        U_L = MF_U_L.arrays()[0];
        U_R = MF_U_R.arrays()[0];

        // State at 2,3,4 will have
        // rho = 1.2
        //  U = 2,    V = 3,    W = -1
        // MX = 2.4, MY = 3.6, MZ = -1.2
        // p = 9, E = 26.4
        // BX = 2    BY = -1.4   BZ = 0.2
        // B_sq = 6
        // p_T = 12
        //
        // v_a_sq = 5
        // c_s_sq = 12
        // c_f_x_sq = (17+sqrt(129))/2
        // c_f_x = 3.765489124

        // State at 3,3,4 will have
        // rho = 3
        //  U = 0.5,  V = -2,  W = 3
        // MX = 1.5, MY = -6, MZ = 9
        //  p = 15,   E = 52.5
        // BX = 1.5  BY = 3   BZ = -2
        // B_sq = 15.25
        // p_T = 22.625
        //
        // v_a_sq = 61/12
        // c_s_sq = 8
        // c_f_x_sq = 12.60742324
        // c_f_x = 3.550693346
        U_L(i - 1, j, k, PUInd::RHO) = 1.2;
        U_L(i - 1, j, k, PUInd::MX)  = 2.4;
        U_L(i - 1, j, k, PUInd::MY)  = 3.6;
        U_L(i - 1, j, k, PUInd::MZ)  = -1.2;
        U_L(i - 1, j, k, PUInd::E)   = 26.4;
        U_L(i - 1, j, k, PUInd::BX)  = 2;
        U_L(i - 1, j, k, PUInd::BY)  = -1.4;
        U_L(i - 1, j, k, PUInd::BZ)  = 0.2;

        U_R(i, j, k, PUInd::RHO) = 3;
        U_R(i, j, k, PUInd::MX)  = 1.5;
        U_R(i, j, k, PUInd::MY)  = -6;
        U_R(i, j, k, PUInd::MZ)  = 9;
        U_R(i, j, k, PUInd::E)   = 52.5;
        U_R(i, j, k, PUInd::BX)  = 1.5;
        U_R(i, j, k, PUInd::BY)  = 3;
        U_R(i, j, k, PUInd::BZ)  = -2;

        // Bogus values in unused cells
        U_L(i, j, k, PUInd::RHO) = 1;
        U_L(i, j, k, PUInd::MX)  = 0;
        U_L(i, j, k, PUInd::MY)  = 0;
        U_L(i, j, k, PUInd::MZ)  = 0;
        U_L(i, j, k, PUInd::E)   = 1;
        U_L(i, j, k, PUInd::BX)  = 0;
        U_L(i, j, k, PUInd::BY)  = 0;
        U_L(i, j, k, PUInd::BZ)  = 0;

        U_R(i - 1, j, k, PUInd::RHO) = 1;
        U_R(i - 1, j, k, PUInd::MX)  = 0;
        U_R(i - 1, j, k, PUInd::MY)  = 0;
        U_R(i - 1, j, k, PUInd::MZ)  = 0;
        U_R(i - 1, j, k, PUInd::E)   = 1;
        U_R(i - 1, j, k, PUInd::BX)  = 0;
        U_R(i - 1, j, k, PUInd::BY)  = 0;
        U_R(i - 1, j, k, PUInd::BZ)  = 0;

        U_L(i + 1, j, k, PUInd::RHO) = 1;
        U_L(i + 1, j, k, PUInd::MX)  = 0;
        U_L(i + 1, j, k, PUInd::MY)  = 0;
        U_L(i + 1, j, k, PUInd::MZ)  = 0;
        U_L(i + 1, j, k, PUInd::E)   = 1;
        U_L(i + 1, j, k, PUInd::BX)  = 0;
        U_L(i + 1, j, k, PUInd::BY)  = 0;
        U_L(i + 1, j, k, PUInd::BZ)  = 0;

        U_R(i + 1, j, k, PUInd::RHO) = 1;
        U_R(i + 1, j, k, PUInd::MX)  = 0;
        U_R(i + 1, j, k, PUInd::MY)  = 0;
        U_R(i + 1, j, k, PUInd::MZ)  = 0;
        U_R(i + 1, j, k, PUInd::E)   = 1;
        U_R(i + 1, j, k, PUInd::BX)  = 0;
        U_R(i + 1, j, k, PUInd::BY)  = 0;
        U_R(i + 1, j, k, PUInd::BZ)  = 0;
    }
};

TEST_F(HLLC_MHDTest, MULTIDIM(Setup))
{
    ASSERT_TRUE(U_L.contains(i, j, k));
    ASSERT_TRUE(U_L.contains(i - 1, j, k));
    ASSERT_TRUE(U_R.contains(i, j, k));
    ASSERT_TRUE(U_R.contains(i - 1, j, k));

    MultiFab ML_Q_L(ba, dm, PQInd::size, 0);
    MultiFab ML_Q_R(ba, dm, PQInd::size, 0);

    const auto &Q_L = ML_Q_L.arrays()[0];
    const auto &Q_R = ML_Q_R.arrays()[0];

    system->UtoQ(domain, Q_L, U_L, h_parm, d_parm);
    system->UtoQ(domain, Q_R, U_R, h_parm, d_parm);
    EXPECT_NEAR(Q_L(i - 1, j, k, PQInd::RHO), 1.2, 1e-10);
    EXPECT_NEAR(Q_L(i - 1, j, k, PQInd::U), 2, 1e-10);
    EXPECT_NEAR(Q_L(i - 1, j, k, PQInd::V), 3, 1e-10);
    EXPECT_NEAR(Q_L(i - 1, j, k, PQInd::W), -1, 1e-10);
    EXPECT_NEAR(Q_L(i - 1, j, k, PQInd::P), 9, 1e-12);
    EXPECT_NEAR(Q_L(i - 1, j, k, PQInd::PTOT), 12, 1e-12);
    EXPECT_NEAR(Q_L(i - 1, j, k, PQInd::BX), 2, 1e-10);
    EXPECT_NEAR(Q_L(i - 1, j, k, PQInd::BY), -1.4, 1e-10);
    EXPECT_NEAR(Q_L(i - 1, j, k, PQInd::BZ), 0.2, 1e-10);

    EXPECT_NEAR(Q_R(i, j, k, PQInd::RHO), 3, 1e-10);
    EXPECT_NEAR(Q_R(i, j, k, PQInd::U), 0.5, 1e-10);
    EXPECT_NEAR(Q_R(i, j, k, PQInd::V), -2, 1e-10);
    EXPECT_NEAR(Q_R(i, j, k, PQInd::W), 3, 1e-10);
    EXPECT_NEAR(Q_R(i, j, k, PQInd::P), 15, 1e-12);
    EXPECT_NEAR(Q_R(i, j, k, PQInd::PTOT), 22.625, 1e-12);
    EXPECT_NEAR(Q_R(i, j, k, PQInd::BX), 1.5, 1e-10);
    EXPECT_NEAR(Q_R(i, j, k, PQInd::BY), 3, 1e-10);
    EXPECT_NEAR(Q_R(i, j, k, PQInd::BZ), -2, 1e-10);
}

TEST_F(HLLC_MHDTest, MULTIDIM(WaveSpeed))
{
    const auto &Q_L = mhd_ctop_noB(i - 1, j, k, U_L, *h_parm);
    const auto &Q_R = mhd_ctop_noB(i, j, k, U_R, *h_parm);
    Real        S_L, S_R, S_star;
    mhd_HLLC_speeds<0>(i, j, k, U_L, U_R, Q_L, Q_R, *h_parm, S_L, S_R, S_star);
    // S_L = min(u_x_L, u_x_R) - max(c_f_L, c_f_R)
    //     = 0.5 - 3.765489124
    //     = -3.265489124
    // S_R = max(u_x_L, u_x_R) + max(c_f_L, c_f_R)
    //     = 2 + 3.765489124
    //     = 5.765489124
    // S_star = 0.3689978539
    EXPECT_NEAR(S_L, -3.265489124, 1e-9);
    EXPECT_NEAR(S_R, 5.765489124, 1e-9);
    EXPECT_NEAR(S_star, 0.3689978539, 1e-9);
}

TEST_F(HLLC_MHDTest, MULTIDIM(FluxFunc))
{
    const auto &Q_L        = mhd_ctop_noB(i - 1, j, k, U_L, *h_parm);
    const auto &Q_R        = mhd_ctop_noB(i, j, k, U_R, *h_parm);
    const auto  mhd_flux_L = mhd_flux<0>(i - 1, j, k, U_L, Q_L);
    const auto  mhd_flux_R = mhd_flux<0>(i, j, k, U_R, Q_R);

    // f_L = 2.4
    //       12.8
    //       10
    //       -2.8
    //       77.6
    //       0
    //       -8.8
    //       2.4

    EXPECT_NEAR(mhd_flux_L[PUInd::RHO], 2.4, 1e-9);
    EXPECT_NEAR(mhd_flux_L[PUInd::MX], 12.8, 1e-9);
    EXPECT_NEAR(mhd_flux_L[PUInd::MY], 10, 1e-9);
    EXPECT_NEAR(mhd_flux_L[PUInd::MZ], -2.8, 1e-9);
    EXPECT_NEAR(mhd_flux_L[PUInd::E], 77.6, 1e-9);
    EXPECT_NEAR(mhd_flux_L[PUInd::BX], 0, 1e-9);
    EXPECT_NEAR(mhd_flux_L[PUInd::BY], -8.8, 1e-9);
    EXPECT_NEAR(mhd_flux_L[PUInd::BZ], 2.4, 1e-9);

    // f_R = 1.5
    //       21.125
    //       -7.5
    //       7.5
    //       54.4375
    //       0
    //       4.5
    //       -5.5

    EXPECT_NEAR(mhd_flux_R[PUInd::RHO], 1.5, 1e-9);
    EXPECT_NEAR(mhd_flux_R[PUInd::MX], 21.125, 1e-9);
    EXPECT_NEAR(mhd_flux_R[PUInd::MY], -7.5, 1e-9);
    EXPECT_NEAR(mhd_flux_R[PUInd::MZ], 7.5, 1e-9);
    EXPECT_NEAR(mhd_flux_R[PUInd::E], 54.4375, 1e-9);
    EXPECT_NEAR(mhd_flux_R[PUInd::BX], 0, 1e-9);
    EXPECT_NEAR(mhd_flux_R[PUInd::BY], 4.5, 1e-9);
    EXPECT_NEAR(mhd_flux_R[PUInd::BZ], -5.5, 1e-9);
}

TEST_F(HLLC_MHDTest, MULTIDIM(B_HLL))
{
    const auto &Q_L        = mhd_ctop_noB(i - 1, j, k, U_L, *h_parm);
    const auto &Q_R        = mhd_ctop_noB(i, j, k, U_R, *h_parm);
    const auto  mhd_flux_L = mhd_flux<0>(i - 1, j, k, U_L, Q_L);
    const auto  mhd_flux_R = mhd_flux<0>(i, j, k, U_R, Q_R);
    Real        S_L, S_R, S_star;
    mhd_HLLC_speeds<0>(i, j, k, U_L, U_R, Q_L, Q_R, *h_parm, S_L, S_R, S_star);

    // U_HLL = (S_R*U_R - S_L*U_L + f_L - f_R) / (S_R - S_L)
    // BX_HLL = 1.680793765
    // BY_HLL = -0.06369380878
    // BZ_HLL = -0.3297406263
    const auto &B_HLL
        = mhd_HLL_B<0>(i, j, k, U_L, U_R, mhd_flux_L, mhd_flux_R, S_L, S_R);
    EXPECT_NEAR(B_HLL[0], 1.680793765, 1e-9);
    EXPECT_NEAR(B_HLL[1], -0.06369380878, 1e-9);
    EXPECT_NEAR(B_HLL[2], -0.3297406263, 1e-9);
}

TEST_F(HLLC_MHDTest, MULTIDIM(HLLC_state))
{
    const auto &Q_L        = mhd_ctop_noB(i - 1, j, k, U_L, *h_parm);
    const auto &Q_R        = mhd_ctop_noB(i, j, k, U_R, *h_parm);
    const auto  mhd_flux_L = mhd_flux<0>(i - 1, j, k, U_L, Q_L);
    const auto  mhd_flux_R = mhd_flux<0>(i, j, k, U_R, Q_R);
    Real        S_L, S_R, S_star;
    mhd_HLLC_speeds<0>(i, j, k, U_L, U_R, Q_L, Q_R, *h_parm, S_L, S_R, S_star);

    // BX_HLL = 1.680793765
    // BY_HLL = -0.06369380878
    // BZ_HLL = -0.3297406263
    const auto &B_HLL
        = mhd_HLL_B<0>(i, j, k, U_L, U_R, mhd_flux_L, mhd_flux_R, S_L, S_R);

    const auto &U_HLLC_L
        = mhd_HLLC_U_star<0>(i - 1, j, k, U_L, Q_L, B_HLL, S_L, S_star);
    EXPECT_EQ(U_HLLC_L[PUInd::BX], B_HLL[0]);
    EXPECT_EQ(U_HLLC_L[PUInd::BY], B_HLL[1]);
    EXPECT_EQ(U_HLLC_L[PUInd::BZ], B_HLL[2]);

    // p_T_HLLC_L = 21.13069655

    // U_HLLC_L = 1.738508622
    //            0.6415059505
    //            5.95646781
    //            -2.001056265
    //            43.28679835

    // HLLC v_L = 0.3689978539
    //            3.426194
    //            -1.15101889
    // B.v_HLLC_L = 0.7815196363
    EXPECT_NEAR(U_HLLC_L[PUInd::RHO], 1.738508622, 1e-9);
    EXPECT_NEAR(U_HLLC_L[PUInd::MX], 0.6415059505, 1e-9);
    EXPECT_NEAR(U_HLLC_L[PUInd::MY], 5.95646781, 1e-9);
    EXPECT_NEAR(U_HLLC_L[PUInd::MZ], -2.001056265, 1e-9);
    EXPECT_NEAR(U_HLLC_L[PUInd::E], 43.28679835, 1e-6); // mfw rounding errors
}

TEST_F(HLLC_MHDTest, MULTIDIM(Final_Flux))
{
    // S_L < 0 < S_star < S_R
    // so f_HLLC = f_L + S_L * (U_HLLC_L - U_L)

    // f_L = 2.4
    //       12.8
    //       10
    //       -2.8
    //       77.6
    //       0
    //       -8.8
    //       2.4

    // U_HLLC_L = 1.738508622
    //            0.6415059505
    //            5.95646781
    //            -2.001056265
    //            43.28679835
    //            1.680793765
    //            -0.06369380878
    //            -0.3297406263

    // f_HLLC = 0.6415059517
    //          18.54234334
    //          2.304979995
    //          -0.1841594789
    //          22.45634365
    //          1.042364489
    //          -13.16369333
    //          4.129862254
    const Box      fbx_decl((IntVect(AMREX_D_DECL(i, j, k))),
                            (IntVect(AMREX_D_DECL(i + 1, j, k))));
    const Box      fbx((IntVect(AMREX_D_DECL(i, j, k))),
                       (IntVect(AMREX_D_DECL(i, j, k))));
    const BoxArray fba(fbx_decl);
    MultiFab       MF_FLUX(fba, dm, PUInd::size, 0);
    const auto    &flux = MF_FLUX.arrays()[0];
    const auto    &dx   = geom.CellSizeArray();
    compute_HLLC_flux_LR<0>(0, fbx, flux, U_L, U_R, dx, 0.01, system, h_parm,
                            d_parm);

    EXPECT_NEAR(flux(i, j, k, PUInd::RHO), 0.6415059517, 1e-8);
    EXPECT_NEAR(flux(i, j, k, PUInd::MX), 18.54234334, 1e-6);
    EXPECT_NEAR(flux(i, j, k, PUInd::MY), 2.304979995, 1e-8);
    EXPECT_NEAR(flux(i, j, k, PUInd::MZ), -0.1841594789, 1e-8);
    EXPECT_NEAR(flux(i, j, k, PUInd::E), 22.45634365, 1e-6); // rounding
    EXPECT_NEAR(flux(i, j, k, PUInd::BX), 1.042364489, 1e-8);
    EXPECT_NEAR(flux(i, j, k, PUInd::BY), -13.16369333, 1e-8);
    EXPECT_NEAR(flux(i, j, k, PUInd::BZ), 4.129862254, 1e-8);
}