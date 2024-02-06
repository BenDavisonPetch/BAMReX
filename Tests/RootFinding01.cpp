#include "Euler/Euler.H"
#include "Euler/RiemannSolver.H"
#include "RootFinding.H"

#include "UnitDomainTest.H"

#include <AMReX_REAL.H>
#include <gtest/gtest.h>

using namespace amrex;

TEST(RootFinding, Polynomials)
{
    std::function<Real(Real)> f       = [](Real x) { return x * x; };
    std::function<Real(Real)> f_deriv = [](Real x) { return 2 * x; };
    Real                      x       = 1.;
    newton_raphson(x, f, f_deriv);

    EXPECT_NEAR(x, 0, 1e-6);

    f       = [](Real x) { return (x - 2) * (x - 2) * (x - 2) - 1; };
    f_deriv = [](Real x) { return 3 * (x - 2) * (x - 2); };
    x       = 10;
    newton_raphson(x, f, f_deriv);

    EXPECT_NEAR(x, 3, 1e-6);
}

TEST_F(ToroTest1, IntermediatePressure)
{
    Real p_interm = euler_RP_interm_p(0, primv_L, primv_R, 1.4, 1.4);
    EXPECT_NEAR(0.30313, p_interm, 0.000005);
}

TEST_F(ToroTest2, IntermediatePressure)
{
    Real p_interm = euler_RP_interm_p(0, primv_L, primv_R, 1.4, 1.4);
    EXPECT_NEAR(0.00189, p_interm, 0.000005);
}

TEST_F(ToroTest3, IntermediatePressure)
{
    Real p_interm = euler_RP_interm_p(0, primv_L, primv_R, 1.4, 1.4);
    EXPECT_NEAR(460.894, p_interm, 0.0005);
}

TEST_F(ToroTest4, IntermediatePressure)
{
    Real p_interm = euler_RP_interm_p(0, primv_L, primv_R, 1.4, 1.4);
    EXPECT_NEAR(46.0950, p_interm, 0.00005);
}

TEST_F(ToroTest5, IntermediatePressure)
{
    Real p_interm = euler_RP_interm_p(0, primv_L, primv_R, 1.4, 1.4);
    EXPECT_NEAR(1691.64, p_interm, 0.01);
}