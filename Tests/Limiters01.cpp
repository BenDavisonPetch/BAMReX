#include <gtest/gtest.h>

#include "Fluxes/Limiters.H"

TEST(Limiters, Minmod) {
    EXPECT_EQ(minmod(-0.4, 0.2), 0);
    EXPECT_EQ(minmod(0.5, -5.6), 0);
    EXPECT_EQ(minmod(0, 0), 0);
    EXPECT_EQ(minmod(0.4, 0), 0);
    EXPECT_EQ(minmod(-0.4, 0), 0);
    EXPECT_EQ(minmod(0, 0.4), 0);
    EXPECT_EQ(minmod(0, -12), 0);
    EXPECT_EQ(minmod(-0.7, -1.5), -0.7);
    EXPECT_EQ(minmod(-12.1, -0.1), -0.1);
    EXPECT_EQ(minmod(5.7, 12), 5.7);
    EXPECT_EQ(minmod(6, 0.1), 0.1);
}