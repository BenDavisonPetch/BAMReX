#include "GFM/GFMFlag.H"

#include "UnitDomainTest.H"

#include <gtest/gtest.h>

#include <bitset>

using namespace amrex;

#if AMREX_SPACEDIM != 2
#error "Test must be 2D"
#endif

class GFMFlagTest : public UnitDomainTest {};

TEST(GFMFlagFunctions, flag_functions) {
    {
        int flag = GFMFlag::ghost | GFMFlag::interface | GFMFlag::fill;
        EXPECT_TRUE(is_ghost(flag));
        EXPECT_TRUE(at_interface(flag));
        EXPECT_TRUE(needs_fill(flag));
    }
    {
        int flag = GFMFlag::ghost | GFMFlag::interface;
        EXPECT_TRUE(is_ghost(flag));
        EXPECT_TRUE(at_interface(flag));
        EXPECT_FALSE(needs_fill(flag));
    }
    {
        int flag = GFMFlag::ghost | GFMFlag::fill;
        EXPECT_TRUE(is_ghost(flag));
        EXPECT_FALSE(at_interface(flag));
        EXPECT_TRUE(needs_fill(flag));
    }
    {
        int flag = GFMFlag::fill | GFMFlag::interface;
        EXPECT_FALSE(is_ghost(flag));
        EXPECT_TRUE(at_interface(flag));
        EXPECT_TRUE(needs_fill(flag));
    }
    {
        int flag = GFMFlag::ghost;
        EXPECT_TRUE(is_ghost(flag));
        EXPECT_FALSE(at_interface(flag));
        EXPECT_FALSE(needs_fill(flag));
    }
    {
        int flag = GFMFlag::interface;
        EXPECT_FALSE(is_ghost(flag));
        EXPECT_TRUE(at_interface(flag));
        EXPECT_FALSE(needs_fill(flag));
    }
    {
        int flag = GFMFlag::fill;
        EXPECT_FALSE(is_ghost(flag));
        EXPECT_FALSE(at_interface(flag));
        EXPECT_TRUE(needs_fill(flag));
    }
}

TEST_F(GFMFlagTest, build_gfm_flags) {
    // setup unit domain test
    ParmParse pp("prob");
    pp.add("adiabatic", 1.4);
    // 5x5 grid
    setup(5);

    MultiFab LS(ba, dm, 1, 2);
    iMultiFab gfm_flags(ba, dm, 1, 0);

    ASSERT_EQ(LS.size(), 1);
    ASSERT_EQ(gfm_flags.size(), 1);
    
    // Init level set
    for (MFIter mfi(LS); mfi.isValid(); ++mfi) {
        const auto& bx = grow(mfi.validbox(), 2);
        const auto& ls = LS.array(mfi);
        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            // body is present for y <= x
            // for LS, +ve -> body, -ve -> fluid
            Real x = (i + 0.5) * dx[0];
            // cells at i==j are not at LS = 0 but LS = small +ve
            Real y = (j + 0.5) * dx[1] - 1e-10;
            Real d = (x-y) / sqrt(2);
            ls(i,j,k) = d;
        });
    }

    build_gfm_flags(gfm_flags, LS);

    // Test flags
    for (MFIter mfi(LS); mfi.isValid(); ++mfi) {
        const auto& bx = mfi.validbox();
        const auto& flagarr = gfm_flags.const_array(mfi);
        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            auto flag = flagarr(i,j,k);
            Print() << "(i,j) = (" << i << "," << j << ")\tflag = " << std::bitset<8>(flag) << std::endl;
            // ghost flag
            if (i >= j)
                EXPECT_TRUE(is_ghost(flag));
            else
                EXPECT_FALSE(is_ghost(flag));

            // interface flag
            if (i == j || j == i+1)
                EXPECT_TRUE(at_interface(flag));
            else
                EXPECT_FALSE(at_interface(flag));

            // fill flag
            if (i >= j && i <= j+GFM_GROW-1)
                EXPECT_TRUE(needs_fill(flag));
            else
                EXPECT_FALSE(needs_fill(flag));
        });
    }
}