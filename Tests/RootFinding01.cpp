#include "System/Euler/Euler.H"
#include "System/Euler/RiemannSolver.H"
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
    EulerExactRiemannSolver<0> rp_solver(primv_L, primv_R, 1.4, 1.4, 1);
    Real                       p_interm = rp_solver.get_p_interm();
    EXPECT_NEAR(exp_p_interm, p_interm, 0.000005);
}

TEST_F(ToroTest2, IntermediatePressure)
{
    EulerExactRiemannSolver<0> rp_solver(primv_L, primv_R, 1.4, 1.4, 1);
    Real                       p_interm = rp_solver.get_p_interm();
    EXPECT_NEAR(exp_p_interm, p_interm, 0.000005);
}

TEST_F(ToroTest3, IntermediatePressure)
{
    EulerExactRiemannSolver<0> rp_solver(primv_L, primv_R, 1.4, 1.4, 1);
    Real                       p_interm = rp_solver.get_p_interm();
    EXPECT_NEAR(exp_p_interm, p_interm, 0.0005);
}

TEST_F(ToroTest4, IntermediatePressure)
{
    EulerExactRiemannSolver<0> rp_solver(primv_L, primv_R, 1.4, 1.4, 1);
    Real                       p_interm = rp_solver.get_p_interm();
    EXPECT_NEAR(exp_p_interm, p_interm, 0.00005);
}

TEST_F(ToroTest5, IntermediatePressure)
{
    EulerExactRiemannSolver<0> rp_solver(primv_L, primv_R, 1.4, 1.4, 1);
    Real                       p_interm = rp_solver.get_p_interm();
    EXPECT_NEAR(exp_p_interm, p_interm, 0.01);
}

TEST_F(ToroTest1Scaled, IntermediatePressure)
{
    EulerExactRiemannSolver<0> rp_solver(primv_L, primv_R, 1.4, 1.4, epsilon);
    Real                       p_interm = rp_solver.get_p_interm();
    EXPECT_NEAR(exp_p_interm, p_interm, 0.000005 / p_0);
}