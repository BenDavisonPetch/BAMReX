#include <gtest/gtest.h>

#include "IMEX/ButcherTableau.H"

TEST(ButcherTableau, EnumFromString) {
    EXPECT_EQ(IMEXButcherTableau::enum_from_string("SP111"), IMEXButcherTableau::SP111);
    EXPECT_EQ(IMEXButcherTableau::enum_from_string("SSP222"), IMEXButcherTableau::SSP222);
    EXPECT_EQ(IMEXButcherTableau::enum_from_string("SASSP322"), IMEXButcherTableau::SASSP322);
    EXPECT_EQ(IMEXButcherTableau::enum_from_string("SASSP332"), IMEXButcherTableau::SASSP332);
    EXPECT_EQ(IMEXButcherTableau::enum_from_string("SSP433"), IMEXButcherTableau::SSP433);
}

void test_basics(const IMEXButcherTableau& tab) {
    // Check sizes
    auto A_imp = tab.get_A_imp();
    ASSERT_EQ(A_imp.size(), tab.n_steps_imp());
    for (int i = 0; i < A_imp.size(); ++i)
        ASSERT_EQ(A_imp[i].size(), tab.n_steps_imp());
    auto A_exp = tab.get_A_exp();
    ASSERT_EQ(A_exp.size(), tab.n_steps_imp());
    for (int i = 0; i < A_exp.size(); ++i)
        ASSERT_EQ(A_exp[i].size(), tab.n_steps_imp());
    auto b_imp = tab.get_b_imp();
    auto b_exp = tab.get_b_exp();
    ASSERT_EQ(b_imp.size(), tab.n_steps_imp());
    ASSERT_EQ(b_exp.size(), tab.n_steps_imp());
    for (int i = 0; i < tab.n_steps_imp(); ++i)
        EXPECT_EQ(b_imp[i], b_exp[i]);

    // Check that row get functions work
    const double mult = 1.5;
    for (int r = 0; r < tab.n_steps_imp(); ++r) {
        auto improw = tab.get_A_imp_row(r, mult);
        auto exprow = tab.get_A_exp_row(r, mult);
        for (int c = 0; c < tab.n_steps_imp(); ++c) {
            EXPECT_NEAR(improw[c], tab.get_A_imp()[r][c]*mult, 1e-12);
            EXPECT_NEAR(exprow[c], tab.get_A_exp()[r][c]*mult, 1e-12);
        }
    }
}

TEST(ButcherTableau, SP111) {
    IMEXButcherTableau tab(IMEXButcherTableau::SP111);
    ASSERT_EQ(tab.n_steps_imp(), 1);
    ASSERT_EQ(tab.n_steps_exp(), 1);
    ASSERT_EQ(tab.order(), 1);

    // Check vector sizes and row recall
    test_basics(tab);

    EXPECT_EQ(tab.get_A_exp()[0][0], 0);
    EXPECT_EQ(tab.get_A_imp()[0][0], 1);

    EXPECT_EQ(tab.get_b_imp()[0], 1);
}

TEST(ButcherTableau, SSP222) {
    IMEXButcherTableau tab(IMEXButcherTableau::SSP222);
    ASSERT_EQ(tab.n_steps_imp(), 2);
    ASSERT_EQ(tab.n_steps_exp(), 2);
    ASSERT_EQ(tab.order(), 2);

    // Check vector sizes and row recall
    test_basics(tab);

    EXPECT_EQ(tab.get_A_exp()[0][0], 0);
    EXPECT_EQ(tab.get_A_exp()[0][1], 0);
    EXPECT_EQ(tab.get_A_exp()[1][0], 1);
    EXPECT_EQ(tab.get_A_exp()[1][1], 0);

    const amrex::Real gamma = 1 - 1. / sqrt(2.);
    EXPECT_NEAR(tab.get_A_imp()[0][0], gamma, 1e-12);
    EXPECT_NEAR(tab.get_A_imp()[0][1], 0, 1e-12);
    EXPECT_NEAR(tab.get_A_imp()[1][0], 1 - 2 * gamma, 1e-12);
    EXPECT_NEAR(tab.get_A_imp()[1][1], gamma, 1e-12);
    
    EXPECT_EQ(tab.get_b_imp()[0], 0.5);
    EXPECT_EQ(tab.get_b_imp()[1], 0.5);
}

TEST(ButcherTableau, SASSP322) {
    IMEXButcherTableau tab(IMEXButcherTableau::SASSP322);
    ASSERT_EQ(tab.n_steps_imp(), 3);
    ASSERT_EQ(tab.n_steps_exp(), 2);
    ASSERT_EQ(tab.order(), 2);

    // Check vector sizes and row recall
    test_basics(tab);

    // A_exp
    EXPECT_EQ(tab.get_A_exp()[0][0], 0);
    EXPECT_EQ(tab.get_A_exp()[0][1], 0);
    EXPECT_EQ(tab.get_A_exp()[0][2], 0);

    EXPECT_EQ(tab.get_A_exp()[1][0], 0);
    EXPECT_EQ(tab.get_A_exp()[1][1], 0);
    EXPECT_EQ(tab.get_A_exp()[1][2], 0);

    EXPECT_EQ(tab.get_A_exp()[2][0], 0);
    EXPECT_EQ(tab.get_A_exp()[2][1], 1);
    EXPECT_EQ(tab.get_A_exp()[2][2], 0);

    // A_imp
    EXPECT_EQ(tab.get_A_imp()[0][0], 0.5);
    EXPECT_EQ(tab.get_A_imp()[0][1], 0);
    EXPECT_EQ(tab.get_A_imp()[0][2], 0);

    EXPECT_EQ(tab.get_A_imp()[1][0], -0.5);
    EXPECT_EQ(tab.get_A_imp()[1][1], 0.5);
    EXPECT_EQ(tab.get_A_imp()[1][2], 0);

    EXPECT_EQ(tab.get_A_imp()[2][0], 0);
    EXPECT_EQ(tab.get_A_imp()[2][1], 0.5);
    EXPECT_EQ(tab.get_A_imp()[2][2], 0.5);

    // b
    EXPECT_EQ(tab.get_b_imp()[0], 0);
    EXPECT_EQ(tab.get_b_imp()[1], 0.5);
    EXPECT_EQ(tab.get_b_imp()[2], 0.5);
}

TEST(ButcherTableau, SASSP332) {
    IMEXButcherTableau tab(IMEXButcherTableau::SASSP332);
    ASSERT_EQ(tab.n_steps_imp(), 3);
    ASSERT_EQ(tab.n_steps_exp(), 3);
    ASSERT_EQ(tab.order(), 2);

    // Check vector sizes and row recall
    test_basics(tab);

    // A_exp
    EXPECT_EQ(tab.get_A_exp()[0][0], 0);
    EXPECT_EQ(tab.get_A_exp()[0][1], 0);
    EXPECT_EQ(tab.get_A_exp()[0][2], 0);

    EXPECT_EQ(tab.get_A_exp()[1][0], 0.5);
    EXPECT_EQ(tab.get_A_exp()[1][1], 0);
    EXPECT_EQ(tab.get_A_exp()[1][2], 0);

    EXPECT_EQ(tab.get_A_exp()[2][0], 0.5);
    EXPECT_EQ(tab.get_A_exp()[2][1], 0.5);
    EXPECT_EQ(tab.get_A_exp()[2][2], 0);

    // A_imp
    EXPECT_EQ(tab.get_A_imp()[0][0], 0.25);
    EXPECT_EQ(tab.get_A_imp()[0][1], 0);
    EXPECT_EQ(tab.get_A_imp()[0][2], 0);

    EXPECT_EQ(tab.get_A_imp()[1][0], 0);
    EXPECT_EQ(tab.get_A_imp()[1][1], 0.25);
    EXPECT_EQ(tab.get_A_imp()[1][2], 0);

    EXPECT_NEAR(tab.get_A_imp()[2][0], 1/(amrex::Real)3, 1e-12);
    EXPECT_NEAR(tab.get_A_imp()[2][1], 1/(amrex::Real)3, 1e-12);
    EXPECT_NEAR(tab.get_A_imp()[2][2], 1/(amrex::Real)3, 1e-12);

    // b
    EXPECT_NEAR(tab.get_b_imp()[0], 1/(amrex::Real)3, 1e-12);
    EXPECT_NEAR(tab.get_b_imp()[1], 1/(amrex::Real)3, 1e-12);
    EXPECT_NEAR(tab.get_b_imp()[2], 1/(amrex::Real)3, 1e-12);
}

TEST(ButcherTableau, SSP433) {
    IMEXButcherTableau tab(IMEXButcherTableau::SSP433);
    ASSERT_EQ(tab.n_steps_imp(), 4);
    ASSERT_EQ(tab.n_steps_exp(), 3);
    ASSERT_EQ(tab.order(), 3);

    // Check vector sizes and row recall
    test_basics(tab);

    // A_exp
    EXPECT_EQ(tab.get_A_exp()[0][0], 0);
    EXPECT_EQ(tab.get_A_exp()[0][1], 0);
    EXPECT_EQ(tab.get_A_exp()[0][2], 0);
    EXPECT_EQ(tab.get_A_exp()[0][3], 0);

    EXPECT_EQ(tab.get_A_exp()[1][0], 0);
    EXPECT_EQ(tab.get_A_exp()[1][1], 0);
    EXPECT_EQ(tab.get_A_exp()[1][2], 0);
    EXPECT_EQ(tab.get_A_exp()[1][3], 0);

    EXPECT_EQ(tab.get_A_exp()[2][0], 0);
    EXPECT_EQ(tab.get_A_exp()[2][1], 1);
    EXPECT_EQ(tab.get_A_exp()[2][2], 0);
    EXPECT_EQ(tab.get_A_exp()[2][3], 0);

    EXPECT_EQ(tab.get_A_exp()[3][0], 0);
    EXPECT_EQ(tab.get_A_exp()[3][1], 0.25);
    EXPECT_EQ(tab.get_A_exp()[3][2], 0.25);
    EXPECT_EQ(tab.get_A_exp()[3][3], 0);

    // A_imp
    const amrex::Real alpha = 0.24169426078821;
    const amrex::Real beta = 0.06042356519705;
    const amrex::Real eta = 0.12915286960590;
    EXPECT_NEAR(tab.get_A_imp()[0][0], alpha, 1e-12);
    EXPECT_EQ(tab.get_A_imp()[0][1], 0);
    EXPECT_EQ(tab.get_A_imp()[0][2], 0);
    EXPECT_EQ(tab.get_A_imp()[0][3], 0);

    EXPECT_NEAR(tab.get_A_imp()[1][0], -alpha, 1e-12);
    EXPECT_NEAR(tab.get_A_imp()[1][1], alpha, 1e-12);
    EXPECT_EQ(tab.get_A_imp()[1][2], 0);
    EXPECT_EQ(tab.get_A_imp()[1][3], 0);

    EXPECT_EQ(tab.get_A_imp()[2][0], 0);
    EXPECT_NEAR(tab.get_A_imp()[2][1], 1-alpha, 1e-12);
    EXPECT_NEAR(tab.get_A_imp()[2][2], alpha, 1e-12);
    EXPECT_EQ(tab.get_A_imp()[2][3], 0);

    EXPECT_NEAR(tab.get_A_imp()[3][0], beta, 1e-12);
    EXPECT_NEAR(tab.get_A_imp()[3][1], eta, 1e-12);
    EXPECT_NEAR(tab.get_A_imp()[3][2], 0.5-beta-eta-alpha, 1e-12);
    EXPECT_NEAR(tab.get_A_imp()[3][3], alpha, 1e-12);

    // b
    EXPECT_EQ(tab.get_b_imp()[0], 0);
    EXPECT_NEAR(tab.get_b_imp()[1], 1/(amrex::Real)6, 1e-12);
    EXPECT_NEAR(tab.get_b_imp()[2], 1/(amrex::Real)6, 1e-12);
    EXPECT_NEAR(tab.get_b_imp()[3], 2/(amrex::Real)3, 1e-12);
}