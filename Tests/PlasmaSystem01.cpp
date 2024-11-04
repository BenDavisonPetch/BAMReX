#include <gtest/gtest.h>

#include "Fluxes/Fluxes.H"
#include "System/Index.H"
#include "UnigridTest.H"

class PlasmaSystemTest : public UnigridTest
{
};

TEST(PlasmaSystem, MULTIDIM(StateSize))
{
    ParmParse pp("prob");
    pp.add("system", (std::string) "plasma");
    System *system = System::NewSystem(pp);
    ASSERT_EQ(system->StateSize(), 8);
    delete system;
}

TEST(PlasmaSystem, MULTIDIM(VariableIndex))
{
    ParmParse pp("prob");
    pp.add("system", (std::string) "plasma");
    System *system = System::NewSystem(pp);
    EXPECT_EQ(system->VariableIndex("density"), PUInd::RHO);
    EXPECT_EQ(system->VariableIndex("mom_x"), PUInd::MX);
    EXPECT_EQ(system->VariableIndex("mom_y"), PUInd::MY);
    EXPECT_EQ(system->VariableIndex("mom_z"), PUInd::MZ);
    EXPECT_EQ(system->VariableIndex("energy"), PUInd::E);
    EXPECT_EQ(system->VariableIndex("B_x"), PUInd::BX);
    EXPECT_EQ(system->VariableIndex("B_y"), PUInd::BY);
    EXPECT_EQ(system->VariableIndex("B_z"), PUInd::BZ);
    EXPECT_EQ(system->VariableIndex("u_x"), PQInd::U);
    EXPECT_EQ(system->VariableIndex("u_y"), PQInd::V);
    EXPECT_EQ(system->VariableIndex("u_z"), PQInd::W);
    EXPECT_EQ(system->VariableIndex("p"), PQInd::P);
    delete system;
}

TEST(PlasmaSystem, MULTIDIM(VariableIsPrimitive))
{
    ParmParse pp("prob");
    pp.add("system", (std::string) "plasma");
    System *system = System::NewSystem(pp);
    EXPECT_FALSE(system->VariableIsPrimitive("density"));
    EXPECT_FALSE(system->VariableIsPrimitive("mom_x"));
    EXPECT_FALSE(system->VariableIsPrimitive("mom_y"));
    EXPECT_FALSE(system->VariableIsPrimitive("mom_z"));
    EXPECT_FALSE(system->VariableIsPrimitive("energy"));
    EXPECT_FALSE(system->VariableIsPrimitive("B_x"));
    EXPECT_FALSE(system->VariableIsPrimitive("B_y"));
    EXPECT_FALSE(system->VariableIsPrimitive("B_z"));
    EXPECT_TRUE(system->VariableIsPrimitive("u_x"));
    EXPECT_TRUE(system->VariableIsPrimitive("u_y"));
    EXPECT_TRUE(system->VariableIsPrimitive("u_z"));
    EXPECT_TRUE(system->VariableIsPrimitive("p"));
    delete system;
}

TEST(PlasmaSystem, MULTIDIM(VariableNames_IndexConsistency))
{
    ParmParse pp("prob");
    pp.add("system", (std::string) "plasma");
    System     *system    = System::NewSystem(pp);
    const auto &var_names = system->VariableNames();
    ASSERT_EQ(var_names.size(), system->StateSize());
    // VariableNames must be in order
    for (int n = 0; n < var_names.size(); ++n)
    {
        EXPECT_EQ(system->VariableIndex(var_names[n]), n);
    }
    delete system;
}

TEST_F(PlasmaSystemTest, MULTIDIM(FillFluxes))
{
    ParmParse pp("prob");
    pp.add("system", (std::string) "plasma");
    pp.add("adiabatic", 1.6);
    pp.add("epsilon", 1);
    IntVect                        dom_lo(AMREX_D_DECL(2, 3, 4));
    IntVect                        dom_hi(AMREX_D_DECL(2, 3, 4));
    GpuArray<Real, AMREX_SPACEDIM> prob_lo{ AMREX_D_DECL(0., 0., 0.) };
    GpuArray<Real, AMREX_SPACEDIM> prob_hi{ AMREX_D_DECL(2., 2., 2.) };
    setup(dom_lo, dom_hi, prob_lo, prob_hi, pp, pp);

    MultiFab    MF_U(ba, dm, PUInd::size, 0);
    MultiFab    MF_FX(ba, dm, PUInd::size, 0);
    const auto &U  = MF_U.arrays()[0];
    const auto &FX = MF_FX.arrays()[0];

#if AMREX_SPACEDIM > 1
    MultiFab    MF_FY(ba, dm, PUInd::size, 0);
    const auto &FY = MF_FY.arrays()[0];
#endif
#if AMREX_SPACEDIM > 2
    MultiFab    MF_FZ(ba, dm, PUInd::size, 0);
    const auto &FZ = MF_FZ.arrays()[0];
#endif

    // State at 2,3,4 will have
    // rho = 1.2
    //  U = 2,    V = 3,    W = -1
    // MX = 2.4, MY = 3.6, MZ = -1.2
    // p = 9, E = 26.125
    // B_x = -1
    // B_y = 2.1
    // B_z = 0.2
    // |B|^2 = 5.45
    // u.B = 4.1
    const int i = 2;
    const int j = (AMREX_SPACEDIM > 1) ? 3 : 0;
    const int k = (AMREX_SPACEDIM > 2) ? 4 : 0;
    ASSERT_TRUE(U.contains(i, j, k));
    U(i, j, k, PUInd::RHO) = 1.2;
    U(i, j, k, PUInd::MX)  = 2.4;
    U(i, j, k, PUInd::MY)  = 3.6;
    U(i, j, k, PUInd::MZ)  = -1.2;
    U(i, j, k, PUInd::E)   = 26.125;
    U(i, j, k, PUInd::BX)  = -1;
    U(i, j, k, PUInd::BY)  = 2.1;
    U(i, j, k, PUInd::BZ)  = 0.2;

    system->FillFluxes(0, 0, domain, FX, U, h_parm, d_parm);
#if AMREX_SPACEDIM > 1
    system->FillFluxes(1, 0, domain, FY, U, h_parm, d_parm);
#endif
#if AMREX_SPACEDIM > 2
    system->FillFluxes(2, 0, domain, FZ, U, h_parm, d_parm);
#endif

    EXPECT_NEAR(FX(i, j, k, PUInd::RHO), 2.4, 1e-12);
    EXPECT_NEAR(FX(i, j, k, PUInd::MX), 15.525, 1e-12);
    EXPECT_NEAR(FX(i, j, k, PUInd::MY), 9.3, 1e-12);
    EXPECT_NEAR(FX(i, j, k, PUInd::MZ), -2.2, 1e-12);
    EXPECT_NEAR(FX(i, j, k, PUInd::E), 79.8, 1e-12);
    EXPECT_NEAR(FX(i, j, k, PUInd::BX), 0, 1e-12);
    EXPECT_NEAR(FX(i, j, k, PUInd::BY), 7.2, 1e-12);
    EXPECT_NEAR(FX(i, j, k, PUInd::BZ), -0.6, 1e-12);

#if AMREX_SPACEDIM > 1
    EXPECT_NEAR(FY(i, j, k, PUInd::RHO), 3.6, 1e-12);
    EXPECT_NEAR(FY(i, j, k, PUInd::MX), 9.3, 1e-12);
    EXPECT_NEAR(FY(i, j, k, PUInd::MY), 18.115, 1e-12);
    EXPECT_NEAR(FY(i, j, k, PUInd::MZ), -4.02, 1e-12);
    EXPECT_NEAR(FY(i, j, k, PUInd::E), 104.94, 1e-12);
    EXPECT_NEAR(FY(i, j, k, PUInd::BX), -7.2, 1e-12);
    EXPECT_NEAR(FY(i, j, k, PUInd::BY), 0, 1e-12);
    EXPECT_NEAR(FY(i, j, k, PUInd::BZ), 2.7, 1e-12);
#endif
#if AMREX_SPACEDIM > 2
    EXPECT_NEAR(FZ(i, j, k, PUInd::RHO), -1.2, 1e-12);
    EXPECT_NEAR(FZ(i, j, k, PUInd::MX), -2.2, 1e-12);
    EXPECT_NEAR(FZ(i, j, k, PUInd::MY), -4.02, 1e-12);
    EXPECT_NEAR(FZ(i, j, k, PUInd::MZ), 12.885, 1e-12);
    EXPECT_NEAR(FZ(i, j, k, PUInd::E), -38.67, 1e-12);
    EXPECT_NEAR(FZ(i, j, k, PUInd::BX), 0.6, 1e-12);
    EXPECT_NEAR(FZ(i, j, k, PUInd::BY), -2.7, 1e-12);
    EXPECT_NEAR(FZ(i, j, k, PUInd::BZ), 0, 1e-12);
#endif
}

TEST_F(PlasmaSystemTest, MULTIDIM(UtoQ))
{
    ParmParse pp("prob");
    pp.add("system", (std::string) "plasma");
    pp.add("adiabatic", 1.6);
    pp.add("epsilon", 1);
    IntVect                        dom_lo(AMREX_D_DECL(2, 3, 4));
    IntVect                        dom_hi(AMREX_D_DECL(2, 3, 4));
    GpuArray<Real, AMREX_SPACEDIM> prob_lo{ AMREX_D_DECL(0., 0., 0.) };
    GpuArray<Real, AMREX_SPACEDIM> prob_hi{ AMREX_D_DECL(2., 2., 2.) };
    setup(dom_lo, dom_hi, prob_lo, prob_hi, pp, pp);

    MultiFab MF_U(ba, dm, PUInd::size, 0);
    MultiFab MF_Q(ba, dm, PQInd::size, 0);

    const auto &U = MF_U.arrays()[0];
    const auto &Q = MF_Q.arrays()[0];

    // State at 2,3,4 will have
    // rho = 1.2
    //  U = 2,    V = 3,    W = -1
    // MX = 2.4, MY = 3.6, MZ = -1.2
    // p = 9, E = 26.125
    // B_x = -1
    // B_y = 2.1
    // B_z = 0.2
    // |B|^2 = 5.45
    // u.B = 4.1
    const int i = 2;
    const int j = (AMREX_SPACEDIM > 1) ? 3 : 0;
    const int k = (AMREX_SPACEDIM > 2) ? 4 : 0;
    ASSERT_TRUE(U.contains(i, j, k));
    U(i, j, k, PUInd::RHO) = 1.2;
    U(i, j, k, PUInd::MX)  = 2.4;
    U(i, j, k, PUInd::MY)  = 3.6;
    U(i, j, k, PUInd::MZ)  = -1.2;
    U(i, j, k, PUInd::E)   = 26.125;
    U(i, j, k, PUInd::BX)  = -1;
    U(i, j, k, PUInd::BY)  = 2.1;
    U(i, j, k, PUInd::BZ)  = 0.2;

    system->UtoQ(domain, Q, U, h_parm, d_parm);

    EXPECT_NEAR(Q(i, j, k, PQInd::RHO), 1.2, 1e-12);
    EXPECT_NEAR(Q(i, j, k, PQInd::U), 2, 1e-12);
    EXPECT_NEAR(Q(i, j, k, PQInd::V), 3, 1e-12);
    EXPECT_NEAR(Q(i, j, k, PQInd::W), -1, 1e-12);
    EXPECT_NEAR(Q(i, j, k, PQInd::P), 9, 1e-12);
    EXPECT_NEAR(Q(i, j, k, PQInd::BX), -1, 1e-12);
    EXPECT_NEAR(Q(i, j, k, PQInd::BY), 2.1, 1e-12);
    EXPECT_NEAR(Q(i, j, k, PQInd::BZ), 0.2, 1e-12);
}

TEST_F(PlasmaSystemTest, MULTIDIM(ExplicitTimeStepDimSplit))
{
    ParmParse pp("prob");
    pp.add("system", (std::string) "plasma");
    pp.add("adiabatic", 1.6);
    pp.add("epsilon", 1);
    IntVect                        dom_lo(AMREX_D_DECL(2, 3, 4));
    IntVect                        dom_hi(AMREX_D_DECL(2, 3, 4));
    GpuArray<Real, AMREX_SPACEDIM> prob_lo{ AMREX_D_DECL(0., 0., 0.) };
    GpuArray<Real, AMREX_SPACEDIM> prob_hi{ AMREX_D_DECL(1., 2., 3.) };
    setup(dom_lo, dom_hi, prob_lo, prob_hi, pp, pp);

    MultiFab MF_U(ba, dm, PUInd::size, 0);
    MultiFab MF_Q(ba, dm, PUInd::size, 0);

    const auto &U = MF_U.arrays()[0];

    // State at 2,3,4 will have
    // rho = 1.2
    //  U = -2,    V = 3,    W = -1
    // MX = -2.4, MY = 3.6, MZ = -1.2
    // p = 9, E = 26.125
    // B_x = -1
    // B_y = 2.1
    // B_z = 0.2
    // |B|^2 = 5.45
    // u.B = 4.1
    const int i = 2;
    const int j = (AMREX_SPACEDIM > 1) ? 3 : 0;
    const int k = (AMREX_SPACEDIM > 2) ? 4 : 0;
    ASSERT_TRUE(U.contains(i, j, k));
    U(i, j, k, PUInd::RHO) = 1.2;
    U(i, j, k, PUInd::MX)  = -2.4;
    U(i, j, k, PUInd::MY)  = 3.6;
    U(i, j, k, PUInd::MZ)  = -1.2;
    U(i, j, k, PUInd::E)   = 26.125;
    U(i, j, k, PUInd::BX)  = -1;
    U(i, j, k, PUInd::BY)  = 2.1;
    U(i, j, k, PUInd::BZ)  = 0.2;

    // c_s_sq = 12
    // v_a_sq = 5.45 / 1.2 = 109/24
    // c_f_x_sq = 15.91325991
    // c_f_y_sq = 13.20101427
    // c_f_z_sq = 16.51744985
    // dx = 1, dy = 2, dz = 3
    // dx / (|u| + c_f_x_sq) = 0.1669688036

    ASSERT_EQ(h_parm->adiabatic, 1.6);
    ASSERT_EQ(h_parm->epsilon, 1);

    ASSERT_EQ(d_parm->adiabatic, 1.6);
    ASSERT_EQ(d_parm->epsilon, 1);

    EXPECT_NEAR(
        system->ExplicitTimeStepDimSplit(0, geom, MF_U, h_parm, d_parm),
        0.1669688036, 1e-8);
}

#if AMREX_SPACEDIM == 3
TEST_F(PlasmaSystemTest, SLICEulerFluxEquivalence)
{
    ParmParse pp("prob");
    pp.add("system", (std::string) "plasma");
    pp.add("adiabatic", 1.6);
    pp.add("epsilon", 1);
    IntVect                        dom_lo(AMREX_D_DECL(1, 3, 4));
    IntVect                        dom_hi(AMREX_D_DECL(5, 3, 4));
    GpuArray<Real, AMREX_SPACEDIM> prob_lo{ AMREX_D_DECL(0., 0., 0.) };
    GpuArray<Real, AMREX_SPACEDIM> prob_hi{ AMREX_D_DECL(0.2, 1., 1.) };
    setup(dom_lo, dom_hi, prob_lo, prob_hi, pp, pp);

    const auto &dx = geom.CellSizeArray();

    Box      flux_bx(IntVect(AMREX_D_DECL(3, 3, 4)),
                     IntVect(AMREX_D_DECL(4, 3, 4)));
    BoxArray flux_ba(flux_bx);
    Box      flux_comp_bx(IntVect(AMREX_D_DECL(3, 3, 4)),
                          IntVect(AMREX_D_DECL(3, 3, 4)));

    MultiFab MF_U(ba, dm, PUInd::size, 0);
    MultiFab MF_FX(flux_ba, dm, PUInd::size, 0);

    const auto &U  = MF_U.arrays()[0];
    const auto &FX = MF_FX.arrays()[0];

    // State at 2,3,4 will have
    // rho = 1.2
    //  U = 2,    V = 3,    W = -1
    // MX = 2.4, MY = 3.6, MZ = -1.2
    // p = 9, E = 23.4
    // B = 0
    const int i = 3; // cell for which we will calculate the flux either side
    const int j = (AMREX_SPACEDIM > 1) ? 3 : 0;
    const int k = (AMREX_SPACEDIM > 2) ? 4 : 0;
    ASSERT_TRUE(U.contains(i, j, k));
    U(i - 2, j, k, PUInd::RHO) = 1.2;
    U(i - 2, j, k, PUInd::MX)  = 2.4;
    U(i - 2, j, k, PUInd::MY)  = 3.6;
    U(i - 2, j, k, PUInd::MZ)  = -1.2;
    U(i - 2, j, k, PUInd::E)   = 23.4;
    U(i - 2, j, k, PUInd::BX)  = 0;
    U(i - 2, j, k, PUInd::BY)  = 0;
    U(i - 2, j, k, PUInd::BZ)  = 0;

    U(i - 1, j, k, PUInd::RHO) = 1.2;
    U(i - 1, j, k, PUInd::MX)  = 2.4;
    U(i - 1, j, k, PUInd::MY)  = 3.6;
    U(i - 1, j, k, PUInd::MZ)  = -1.2;
    U(i - 1, j, k, PUInd::E)   = 23.4;
    U(i - 1, j, k, PUInd::BX)  = 0;
    U(i - 1, j, k, PUInd::BY)  = 0;
    U(i - 1, j, k, PUInd::BZ)  = 0;

    U(i, j, k, PUInd::RHO) = 2.1;
    U(i, j, k, PUInd::MX)  = -3;
    U(i, j, k, PUInd::MY)  = 0;
    U(i, j, k, PUInd::MZ)  = -1.2;
    U(i, j, k, PUInd::E)   = 30;
    U(i, j, k, PUInd::BX)  = 0;
    U(i, j, k, PUInd::BY)  = 0;
    U(i, j, k, PUInd::BZ)  = 0;

    U(i + 1, j, k, PUInd::RHO) = 0.1;
    U(i + 1, j, k, PUInd::MX)  = 2;
    U(i + 1, j, k, PUInd::MY)  = -0.1;
    U(i + 1, j, k, PUInd::MZ)  = 0;
    U(i + 1, j, k, PUInd::E)   = 25;
    U(i + 1, j, k, PUInd::BX)  = 0;
    U(i + 1, j, k, PUInd::BY)  = 0;
    U(i + 1, j, k, PUInd::BZ)  = 0;

    U(i + 2, j, k, PUInd::RHO) = 0.1;
    U(i + 2, j, k, PUInd::MX)  = 2;
    U(i + 2, j, k, PUInd::MY)  = -0.1;
    U(i + 2, j, k, PUInd::MZ)  = 0;
    U(i + 2, j, k, PUInd::E)   = 25;
    U(i + 2, j, k, PUInd::BX)  = 0;
    U(i + 2, j, k, PUInd::BY)  = 0;
    U(i + 2, j, k, PUInd::BZ)  = 0;

    // now make a fluid system
    pp.add("system", (std::string) "fluid");
    System     *fluidsys = System::NewSystem(pp);
    MultiFab    MF_FX_EULER(flux_ba, dm, UInd::size, 0);
    const auto &FX_EULER = MF_FX_EULER.arrays()[0];

    compute_SLIC_flux(0, 0, flux_comp_bx, FX, U, dx, 0.01, system, h_parm,
                      d_parm);
    compute_SLIC_flux(0, 0, flux_comp_bx, FX_EULER, U, dx, 0.01, fluidsys,
                      h_parm, d_parm);

    For(flux_bx, UInd::size,
        [=](int i, int j, int k, int n)
        { EXPECT_NEAR(FX(i, j, k, n), FX_EULER(i, j, k, n), 1e-12); });

    delete fluidsys;
}
#endif