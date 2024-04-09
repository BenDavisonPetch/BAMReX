#include "Euler/Euler.H"
#include "Euler/RiemannSolver.H"

#include "UnitDomainTest.H"

#include <AMReX_REAL.H>
#include <gtest/gtest.h>

using namespace amrex;

TEST_F(ToroTest1, ExactRiemann)
{
    setup(100);
    const amrex::Real adiabatic = 1.4;
    for (MFIter mfi(mf); mfi.isValid(); ++mfi)
    {
        const Box  &bx  = mfi.validbox();
        const auto &arr = mf.array(mfi);

        std::cout << "About to compute exact riemann problem" << std::endl;

        compute_exact_RP_solution(0, time, bx, dx, prob_lo, 0.5, primv_L,
                                  primv_R, adiabatic, adiabatic, arr);

        std::cout << "Done" << std::endl;

        const auto interm_R_idx = idx_at(0.8);
        const auto interm_L_idx = idx_at(0.6);
        const auto R_idx        = idx_at(0.95);
        const auto L_idx        = idx_at(0.05);

        std::cout << "Computed indices" << std::endl;

        const auto primv_interm_R
            = primv_from_consv(arr, adiabatic, interm_R_idx[0], 0, 0);
        const auto primv_interm_L
            = primv_from_consv(arr, adiabatic, interm_L_idx[0], 0, 0);
        const auto primv_R_new
            = primv_from_consv(arr, adiabatic, R_idx[0], 0, 0);
        const auto primv_L_new
            = primv_from_consv(arr, adiabatic, L_idx[0], 0, 0);

        std::cout << "Compute primitive values" << std::endl;
        std::cout << "Comparing..." << std::endl;

        for (int n = 0; n < AmrLevelAdv::NUM_STATE; ++n)
        {
            EXPECT_NEAR(primv_L_new[n], primv_L[n],
                        std::fabs(primv_L[n] * 1e-12));
            EXPECT_NEAR(primv_R_new[n], primv_R[n],
                        std::fabs(primv_R[n] * 1e-12));
            // Exact values only given to 5d.p.
            EXPECT_NEAR(primv_interm_R[n], expected_interm_R[n], 0.00001);
            EXPECT_NEAR(primv_interm_L[n], expected_interm_L[n], 0.00001);
        }
    }
}

TEST_F(ToroTest2, ExactRiemann)
{
    setup(100);
    const amrex::Real adiabatic = 1.4;
    for (MFIter mfi(mf); mfi.isValid(); ++mfi)
    {
        const Box  &bx  = mfi.validbox();
        const auto &arr = mf.array(mfi);

        compute_exact_RP_solution(0, time, bx, dx, prob_lo, 0.5, primv_L,
                                  primv_R, adiabatic, adiabatic, arr);

        const auto interm_R_idx = idx_at(0.51);
        const auto interm_L_idx = idx_at(0.49);
        const auto R_idx        = idx_at(0.95);
        const auto L_idx        = idx_at(0.05);

        const auto primv_interm_R
            = primv_from_consv(arr, adiabatic, interm_R_idx[0], 0, 0);
        const auto primv_interm_L
            = primv_from_consv(arr, adiabatic, interm_L_idx[0], 0, 0);
        const auto primv_R_new
            = primv_from_consv(arr, adiabatic, R_idx[0], 0, 0);
        const auto primv_L_new
            = primv_from_consv(arr, adiabatic, L_idx[0], 0, 0);

        for (int n = 0; n < AmrLevelAdv::NUM_STATE; ++n)
        {
            EXPECT_NEAR(primv_L_new[n], primv_L[n],
                        std::fabs(primv_L[n] * 1e-12));
            EXPECT_NEAR(primv_R_new[n], primv_R[n],
                        std::fabs(primv_R[n] * 1e-12));
            // Exact values given to 5 d.p. only
            EXPECT_NEAR(primv_interm_R[n], expected_interm_R[n], 0.00001);
            EXPECT_NEAR(primv_interm_L[n], expected_interm_L[n], 0.00001);
        }
    }
}

TEST_F(ToroTest3, ExactRiemann)
{
    setup(100);
    const amrex::Real adiabatic = 1.4;
    for (MFIter mfi(mf); mfi.isValid(); ++mfi)
    {
        const Box  &bx  = mfi.validbox();
        const auto &arr = mf.array(mfi);

        compute_exact_RP_solution(0, time, bx, dx, prob_lo, 0.5, primv_L,
                                  primv_R, adiabatic, adiabatic, arr);

        const auto interm_R_idx = idx_at(0.75);
        const auto interm_L_idx = idx_at(0.4);
        const auto R_idx        = idx_at(0.8);
        const auto L_idx        = idx_at(0.05);

        const auto primv_interm_R
            = primv_from_consv(arr, adiabatic, interm_R_idx[0], 0, 0);
        const auto primv_interm_L
            = primv_from_consv(arr, adiabatic, interm_L_idx[0], 0, 0);
        const auto primv_R_new
            = primv_from_consv(arr, adiabatic, R_idx[0], 0, 0);
        const auto primv_L_new
            = primv_from_consv(arr, adiabatic, L_idx[0], 0, 0);

        for (int n = 0; n < AmrLevelAdv::NUM_STATE; ++n)
        {
            EXPECT_NEAR(primv_L_new[n], primv_L[n],
                        std::fabs(primv_L[n] * 1e-12));
            EXPECT_NEAR(primv_R_new[n], primv_R[n],
                        std::fabs(primv_R[n] * 1e-12));
            // Because magnitudes of exact values differ need to take a
            // relative tolerance
            EXPECT_NEAR(primv_interm_R[n], expected_interm_R[n],
                        expected_interm_R[n] * 1e-5);
            EXPECT_NEAR(primv_interm_L[n], expected_interm_L[n],
                        expected_interm_L[n] * 1e-5);
        }
    }
}

TEST_F(ToroTest4, ExactRiemann)
{
    setup(100);
    const amrex::Real adiabatic = 1.4;
    for (MFIter mfi(mf); mfi.isValid(); ++mfi)
    {
        const Box  &bx  = mfi.validbox();
        const auto &arr = mf.array(mfi);

        compute_exact_RP_solution(0, time, bx, dx, prob_lo, 0.5, primv_L,
                                  primv_R, adiabatic, adiabatic, arr);

        const auto interm_R_idx = idx_at(0.5);
        const auto interm_L_idx = idx_at(0.25);
        const auto R_idx        = idx_at(0.95);
        const auto L_idx        = idx_at(0.2);

        const auto primv_interm_R
            = primv_from_consv(arr, adiabatic, interm_R_idx[0], 0, 0);
        const auto primv_interm_L
            = primv_from_consv(arr, adiabatic, interm_L_idx[0], 0, 0);
        const auto primv_R_new
            = primv_from_consv(arr, adiabatic, R_idx[0], 0, 0);
        const auto primv_L_new
            = primv_from_consv(arr, adiabatic, L_idx[0], 0, 0);

        for (int n = 0; n < AmrLevelAdv::NUM_STATE; ++n)
        {
            EXPECT_NEAR(primv_L_new[n], primv_L[n],
                        std::fabs(primv_L[n] * 1e-12));
            EXPECT_NEAR(primv_R_new[n], primv_R[n],
                        std::fabs(primv_R[n] * 1e-12));
            // Exact solution of pressure given to only 4 d.p.
            EXPECT_NEAR(primv_interm_R[n], expected_interm_R[n], 0.0001);
            EXPECT_NEAR(primv_interm_L[n], expected_interm_L[n], 0.0001);
        }
    }
}

TEST_F(ToroTest5, ExactRiemann)
{
    setup(100);
    const amrex::Real adiabatic = 1.4;
    for (MFIter mfi(mf); mfi.isValid(); ++mfi)
    {
        const Box  &bx  = mfi.validbox();
        const auto &arr = mf.array(mfi);

        compute_exact_RP_solution(0, time, bx, dx, prob_lo, 0.5, primv_L,
                                  primv_R, adiabatic, adiabatic, arr);

        const auto interm_R_idx = idx_at(0.85);
        const auto interm_L_idx = idx_at(0.7);
        const auto R_idx        = idx_at(0.95);
        const auto L_idx        = idx_at(0.4);

        const auto primv_interm_R
            = primv_from_consv(arr, adiabatic, interm_R_idx[0], 0, 0);
        const auto primv_interm_L
            = primv_from_consv(arr, adiabatic, interm_L_idx[0], 0, 0);
        const auto primv_R_new
            = primv_from_consv(arr, adiabatic, R_idx[0], 0, 0);
        const auto primv_L_new
            = primv_from_consv(arr, adiabatic, L_idx[0], 0, 0);

        for (int n = 0; n < AmrLevelAdv::NUM_STATE; ++n)
        {
            EXPECT_NEAR(primv_L_new[n], primv_L[n],
                        std::fabs(primv_L[n] * 1e-12));
            EXPECT_NEAR(primv_R_new[n], primv_R[n],
                        std::fabs(primv_R[n] * 1e-12));

            EXPECT_NEAR(primv_interm_R[n], expected_interm_R[n],
                        expected_interm_R[n] * 1e-5);
            EXPECT_NEAR(primv_interm_L[n], expected_interm_L[n],
                        expected_interm_L[n] * 1e-5);
        }
    }
}