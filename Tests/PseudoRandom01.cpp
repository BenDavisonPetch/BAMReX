#include <gtest/gtest.h>

#include "RCM/PseudoRandom.H"

TEST(PseudoRandom, Sequence21)
{
    PseudoRandom<2, 1>  random;
    std::vector<double> expected{ 0.5,   0.25,  0.75,   0.125,  0.625,
                                  0.375, 0.875, 0.0625, 0.5625, 0.3125 };
    for (std::size_t i = 0; i < expected.size(); ++i)
    {
        double rand = random.random();
        EXPECT_NEAR(rand, expected[i], 1e-12);
    }
}

TEST(PseudoRandom, Sequence53)
{
    PseudoRandom<5, 3>  random;
    std::vector<double> expected{ 0.6,  0.2,  0.8,  0.4,  0.12,
                                  0.72, 0.32, 0.92, 0.52, 0.04 };
    for (std::size_t i = 0; i < expected.size(); ++i)
    {
        double rand = random.random();
        EXPECT_NEAR(rand, expected[i], 1e-12);
    }
}