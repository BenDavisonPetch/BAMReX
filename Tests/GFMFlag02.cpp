#include "GFM/GFMFlag.H"

#include "UnitDomainTest.H"

#include <gtest/gtest.h>

#include <bitset>

using namespace amrex;

#if AMREX_SPACEDIM != 2
#error "Test must be 2D"
#endif

class GFMFlagTest : public UnitDomainTest
{
};

TEST_F(GFMFlagTest, staircase)
{
    // setup unit domain test
    ParmParse pp("prob");
    pp.add("adiabatic", 1.4);
    // 5x5 grid
    setup(5);

    MultiFab  LS(ba, dm, 1, 3);
    iMultiFab gfm_flags(ba, dm, 1, 0);

    ASSERT_EQ(LS.size(), 1);
    ASSERT_EQ(gfm_flags.size(), 1);

    // Init level set
    for (MFIter mfi(LS); mfi.isValid(); ++mfi)
    {
        const auto &bx = grow(mfi.validbox(), 2);
        const auto &ls = LS.array(mfi);
        ParallelFor(bx,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k)
                    {
                        // body is a staircase
                        // . . . . .
                        // . . . x x
                        // . . . x x
                        // . x x x x
                        // . x x x x

                        // for LS, +ve -> body, -ve -> fluid
                        // body if j < ((int)(i+1)/2)*2
                        ls(i, j, k) = (j < ((int)(i + 1) / 2) * 2) ? 1 : -1;
                    });
    }

    GFMFillInfo fill_info;
    fill_info.stencil = 3;
    build_gfm_flags(gfm_flags, LS, fill_info);

    // Test flags
    for (MFIter mfi(LS); mfi.isValid(); ++mfi)
    {
        const auto &bx      = mfi.validbox();
        const auto &flagarr = gfm_flags.const_array(mfi);
        ParallelFor(bx,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k)
                    {
                        auto flag = flagarr(i, j, k);
                        Print() << "(i,j) = (" << i << "," << j
                                << ")\tflag = " << std::bitset<8>(flag)
                                << std::endl;
                        // ghost flag
                        if (j < ((int)(i + 1) / 2) * 2)
                            EXPECT_TRUE(is_ghost(flag));
                        else
                            EXPECT_FALSE(is_ghost(flag));

                        // interface flag
                        if (i == j || j == i + 1 || j == i - 1)
                            EXPECT_TRUE(at_interface(flag));
                        else
                            EXPECT_FALSE(at_interface(flag));

                        // fill flag
                        if ((i == 4 && j == 0) || !is_ghost(flag))
                            EXPECT_FALSE(needs_fill(flag));
                        else
                            EXPECT_TRUE(needs_fill(flag));
                    });
    }
}

TEST_F(GFMFlagTest, staircase_diagonal)
{
    // setup unit domain test
    ParmParse pp("prob");
    pp.add("adiabatic", 1.4);
    // 5x5 grid
    setup(5);

    MultiFab  LS(ba, dm, 1, 3);
    iMultiFab gfm_flags(ba, dm, 1, 0);

    ASSERT_EQ(LS.size(), 1);
    ASSERT_EQ(gfm_flags.size(), 1);

    // Init level set
    for (MFIter mfi(LS); mfi.isValid(); ++mfi)
    {
        const auto &bx = grow(mfi.validbox(), 2);
        const auto &ls = LS.array(mfi);
        ParallelFor(bx,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k)
                    {
                        // body is a staircase
                        // . . . . .
                        // . . . x x
                        // . . . x x
                        // . x x x x
                        // . x x x x

                        // for LS, +ve -> body, -ve -> fluid
                        // body if j < ((int)(i+1)/2)*2
                        ls(i, j, k) = (j < ((int)(i + 1) / 2) * 2) ? 1 : -1;
                    });
    }

    GFMFillInfo fill_info;
    fill_info.stencil        = 2;
    fill_info.fill_diagonals = true;
    build_gfm_flags(gfm_flags, LS, fill_info);

    // Test flags
    for (MFIter mfi(LS); mfi.isValid(); ++mfi)
    {
        const auto &bx      = mfi.validbox();
        const auto &flagarr = gfm_flags.const_array(mfi);
        ParallelFor(bx,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k)
                    {
                        auto flag = flagarr(i, j, k);
                        Print() << "(i,j) = (" << i << "," << j
                                << ")\tflag = " << std::bitset<8>(flag)
                                << std::endl;
                        // ghost flag
                        if (j < ((int)(i + 1) / 2) * 2)
                            EXPECT_TRUE(is_ghost(flag));
                        else
                            EXPECT_FALSE(is_ghost(flag));

                        // interface flag
                        if (i == j || j == i + 1 || j == i - 1)
                            EXPECT_TRUE(at_interface(flag));
                        else
                            EXPECT_FALSE(at_interface(flag));

                        // fill flag
                        if (i >= j + 3 || !is_ghost(flag))
                            EXPECT_FALSE(needs_fill(flag));
                        else
                            EXPECT_TRUE(needs_fill(flag));
                    });
    }
}