/**
 * Tests for bi(tri)-linear interpolation for GFM
 */

#include "GFM/Interp_K.H"

#include <AMReX_MultiFab.H>

#include <gtest/gtest.h>

using namespace amrex;

class InterpTest : public testing::Test
{
  protected:
    constexpr static int ncells_dst = 5;
    constexpr static int ncomp      = 2;
    IntVect              dst_dom_lo, dst_dom_hi, src_dom_lo, src_dom_hi;
    Box                  domain, domain_src;
    BoxArray             ba, ba_src;
    DistributionMapping  dm, dm_src;
    MultiFab             src, dst;
    GpuArray<Real, AMREX_SPACEDIM> prob_lo, prob_hi;
    RealBox                        real_box;
    Geometry                       geom;
    GpuArray<Real, AMREX_SPACEDIM> dx;
    RealVect                       src_bottomleft;

    InterpTest()
    {
        // Multifab to interpolate (we interpolate in box between cells (1,2,3)
        // and (2,3,4))
        src_dom_lo = IntVect(AMREX_D_DECL(0, 0, 0));
        src_dom_hi = IntVect(AMREX_D_DECL(3, 4, 5));
        domain_src = Box(src_dom_lo, src_dom_hi);
        ba_src     = BoxArray(domain_src);
        dm_src     = DistributionMapping(ba_src);
        src.define(ba_src, dm_src, ncomp, 0);
        src.setVal(1000);

        dst_dom_lo = IntVect(AMREX_D_DECL(0, 0, 0));
        dst_dom_hi = IntVect(
            AMREX_D_DECL(ncells_dst - 1, ncells_dst - 1, ncells_dst - 1));
        domain = Box(dst_dom_lo, dst_dom_hi);
        ba     = BoxArray(domain);
        dm     = DistributionMapping(ba);
        dst.define(ba, dm, ncomp, 0);
        dst.setVal(1000);

        // this is geometry for src
        // dx = 0.5
        // dy = 0.3
        // dz = 0.1
        prob_lo = { AMREX_D_DECL(0.0, 0.0, 0.0) };
        prob_hi = { AMREX_D_DECL(2, 1.5, 0.6) };
        real_box
            = RealBox(AMREX_D_DECL(0.0, 0.0, 0.0), AMREX_D_DECL(2, 1.5, 0.6));
        geom           = Geometry(domain_src, &real_box);
        dx             = geom.CellSizeArray();
        src_bottomleft = RealVect(AMREX_D_DECL(
            (1 + 0.5) * dx[0], (2 + 0.5) * dx[1], (3 + 0.5) * dx[2]));
    }
};

#if AMREX_SPACEDIM == 1
TEST_F(InterpTest, Bilinear1D)
{
    ASSERT_EQ(dx[0], 0.5);
    ASSERT_EQ(src_bottomleft[0], 0.75);

    // interpolating between cell 1 and 2
    const auto &dstarr = dst.arrays()[0];
    const auto &srcarr = src.arrays()[0];

    srcarr(1, 0, 0, 0) = 1;
    srcarr(2, 0, 0, 0) = 2;

    srcarr(1, 0, 0, 1) = -2;
    srcarr(2, 0, 0, 1) = -4;

    for (int i = 0; i < ncells_dst; ++i)
    {
        RealVect offset((Real)i / (ncells_dst - 1) * dx[0]);
        RealVect p = src_bottomleft + offset;
        bilinear_interp(p, srcarr, dstarr, i, 0, 0, dx, ncomp);
    }

    EXPECT_NEAR(dstarr(0, 0, 0, 0), 1, 1e-12);
    EXPECT_NEAR(dstarr(1, 0, 0, 0), 1.25, 1e-12);
    EXPECT_NEAR(dstarr(2, 0, 0, 0), 1.5, 1e-12);
    EXPECT_NEAR(dstarr(3, 0, 0, 0), 1.75, 1e-12);
    EXPECT_NEAR(dstarr(4, 0, 0, 0), 2, 1e-12);

    EXPECT_NEAR(dstarr(0, 0, 0, 1), -2, 1e-12);
    EXPECT_NEAR(dstarr(1, 0, 0, 1), -2.5, 1e-12);
    EXPECT_NEAR(dstarr(2, 0, 0, 1), -3, 1e-12);
    EXPECT_NEAR(dstarr(3, 0, 0, 1), -3.5, 1e-12);
    EXPECT_NEAR(dstarr(4, 0, 0, 1), -4, 1e-12);
}
#endif

#if AMREX_SPACEDIM == 2
TEST_F(InterpTest, Bilinear2D)
{
    ASSERT_EQ(dx[0], 0.5);
    ASSERT_EQ(dx[1], 0.3);

    ASSERT_EQ(src_bottomleft[0], 0.75);
    ASSERT_EQ(src_bottomleft[1], 0.75);

    // interpolating between cell 1,2 and 2,3
    const auto &dstarr = dst.arrays()[0];
    const auto &srcarr = src.arrays()[0];

    srcarr(1, 2, 0, 0) = 1;
    srcarr(2, 2, 0, 0) = 2;
    srcarr(1, 3, 0, 0) = 3;
    srcarr(2, 3, 0, 0) = 4;

    srcarr(1, 2, 0, 1) = -1;
    srcarr(2, 2, 0, 1) = -2;
    srcarr(1, 3, 0, 1) = -3;
    srcarr(2, 3, 0, 1) = -4;

    for (int i = 0; i < ncells_dst; ++i)
    {
        for (int j = 0; j < ncells_dst; ++j)
        {
            RealVect offset((Real)i / (ncells_dst - 1) * dx[0],
                            (Real)j / (ncells_dst - 1) * dx[1]);
            RealVect p = src_bottomleft + offset;
            bilinear_interp(p, srcarr, dstarr, i, j, 0, dx, ncomp);
        }
    }

    // along bottom edge
    EXPECT_NEAR(dstarr(0, 0, 0, 0), 1, 1e-12);
    EXPECT_NEAR(dstarr(1, 0, 0, 0), 1.25, 1e-12);
    EXPECT_NEAR(dstarr(2, 0, 0, 0), 1.5, 1e-12);
    EXPECT_NEAR(dstarr(3, 0, 0, 0), 1.75, 1e-12);
    EXPECT_NEAR(dstarr(4, 0, 0, 0), 2, 1e-12);

    // up left edge
    EXPECT_NEAR(dstarr(0, 0, 0, 0), 1, 1e-12);
    EXPECT_NEAR(dstarr(0, 1, 0, 0), 1.5, 1e-12);
    EXPECT_NEAR(dstarr(0, 2, 0, 0), 2, 1e-12);
    EXPECT_NEAR(dstarr(0, 3, 0, 0), 2.5, 1e-12);
    EXPECT_NEAR(dstarr(0, 4, 0, 0), 3, 1e-12);

    // along top edge
    EXPECT_NEAR(dstarr(0, 4, 0, 0), 3, 1e-12);
    EXPECT_NEAR(dstarr(1, 4, 0, 0), 3.25, 1e-12);
    EXPECT_NEAR(dstarr(2, 4, 0, 0), 3.5, 1e-12);
    EXPECT_NEAR(dstarr(3, 4, 0, 0), 3.75, 1e-12);
    EXPECT_NEAR(dstarr(4, 4, 0, 0), 4, 1e-12);

    // up right edge
    EXPECT_NEAR(dstarr(4, 0, 0, 0), 2, 1e-12);
    EXPECT_NEAR(dstarr(4, 1, 0, 0), 2.5, 1e-12);
    EXPECT_NEAR(dstarr(4, 2, 0, 0), 3, 1e-12);
    EXPECT_NEAR(dstarr(4, 3, 0, 0), 3.5, 1e-12);
    EXPECT_NEAR(dstarr(4, 4, 0, 0), 4, 1e-12);

    // up middle
    EXPECT_NEAR(dstarr(2, 0, 0, 0), 1.5, 1e-12);
    EXPECT_NEAR(dstarr(2, 1, 0, 0), 2, 1e-12);
    EXPECT_NEAR(dstarr(2, 2, 0, 0), 2.5, 1e-12);
    EXPECT_NEAR(dstarr(2, 3, 0, 0), 3, 1e-12);
    EXPECT_NEAR(dstarr(2, 4, 0, 0), 3.5, 1e-12);

    // across middle
    EXPECT_NEAR(dstarr(0, 2, 0, 0), 2, 1e-12);
    EXPECT_NEAR(dstarr(1, 2, 0, 0), 2.25, 1e-12);
    EXPECT_NEAR(dstarr(2, 2, 0, 0), 2.5, 1e-12);
    EXPECT_NEAR(dstarr(3, 2, 0, 0), 2.75, 1e-12);
    EXPECT_NEAR(dstarr(4, 2, 0, 0), 3, 1e-12);

    //
    // Check second component!
    //
    // along bottom edge
    EXPECT_NEAR(dstarr(0, 0, 0, 1), -1, 1e-12);
    EXPECT_NEAR(dstarr(1, 0, 0, 1), -1.25, 1e-12);
    EXPECT_NEAR(dstarr(2, 0, 0, 1), -1.5, 1e-12);
    EXPECT_NEAR(dstarr(3, 0, 0, 1), -1.75, 1e-12);
    EXPECT_NEAR(dstarr(4, 0, 0, 1), -2, 1e-12);

    // up left edge
    EXPECT_NEAR(dstarr(0, 0, 0, 1), -1, 1e-12);
    EXPECT_NEAR(dstarr(0, 1, 0, 1), -1.5, 1e-12);
    EXPECT_NEAR(dstarr(0, 2, 0, 1), -2, 1e-12);
    EXPECT_NEAR(dstarr(0, 3, 0, 1), -2.5, 1e-12);
    EXPECT_NEAR(dstarr(0, 4, 0, 1), -3, 1e-12);

    // along top edge
    EXPECT_NEAR(dstarr(0, 4, 0, 1), -3, 1e-12);
    EXPECT_NEAR(dstarr(1, 4, 0, 1), -3.25, 1e-12);
    EXPECT_NEAR(dstarr(2, 4, 0, 1), -3.5, 1e-12);
    EXPECT_NEAR(dstarr(3, 4, 0, 1), -3.75, 1e-12);
    EXPECT_NEAR(dstarr(4, 4, 0, 1), -4, 1e-12);

    // up right edge
    EXPECT_NEAR(dstarr(4, 0, 0, 1), -2, 1e-12);
    EXPECT_NEAR(dstarr(4, 1, 0, 1), -2.5, 1e-12);
    EXPECT_NEAR(dstarr(4, 2, 0, 1), -3, 1e-12);
    EXPECT_NEAR(dstarr(4, 3, 0, 1), -3.5, 1e-12);
    EXPECT_NEAR(dstarr(4, 4, 0, 1), -4, 1e-12);

    // up middle
    EXPECT_NEAR(dstarr(2, 0, 0, 1), -1.5, 1e-12);
    EXPECT_NEAR(dstarr(2, 1, 0, 1), -2, 1e-12);
    EXPECT_NEAR(dstarr(2, 2, 0, 1), -2.5, 1e-12);
    EXPECT_NEAR(dstarr(2, 3, 0, 1), -3, 1e-12);
    EXPECT_NEAR(dstarr(2, 4, 0, 1), -3.5, 1e-12);

    // across middle
    EXPECT_NEAR(dstarr(0, 2, 0, 1), -2, 1e-12);
    EXPECT_NEAR(dstarr(1, 2, 0, 1), -2.25, 1e-12);
    EXPECT_NEAR(dstarr(2, 2, 0, 1), -2.5, 1e-12);
    EXPECT_NEAR(dstarr(3, 2, 0, 1), -2.75, 1e-12);
    EXPECT_NEAR(dstarr(4, 2, 0, 1), -3, 1e-12);
}
#endif

#if AMREX_SPACEDIM == 3
TEST_F(InterpTest, Bilinear3D)
{
    ASSERT_EQ(dx[0], 0.5);
    ASSERT_EQ(dx[1], 0.3);
    ASSERT_NEAR(dx[2], 0.1, 1e-12);

    ASSERT_EQ(src_bottomleft[0], 0.75);
    ASSERT_EQ(src_bottomleft[1], 0.75);
    ASSERT_EQ(src_bottomleft[2], 0.35);

    // interpolating between cell 1,2,3 and 2,3,4
    const auto &dstarr = dst.arrays()[0];
    const auto &srcarr = src.arrays()[0];

    srcarr(1, 2, 3, 0) = 1;
    srcarr(2, 2, 3, 0) = 2;
    srcarr(1, 3, 3, 0) = 3;
    srcarr(2, 3, 3, 0) = 4;

    srcarr(1, 2, 4, 0) = 5;
    srcarr(2, 2, 4, 0) = 6;
    srcarr(1, 3, 4, 0) = 7;
    srcarr(2, 3, 4, 0) = 8;

    for (int i = 0; i < ncells_dst; ++i)
        for (int j = 0; j < ncells_dst; ++j)
            for (int k = 0; k < ncells_dst; ++k)
            {
                RealVect offset((Real)i / (ncells_dst - 1) * dx[0],
                                (Real)j / (ncells_dst - 1) * dx[1],
                                (Real)k / (ncells_dst - 1) * dx[2]);
                RealVect p = src_bottomleft + offset;
                bilinear_interp(p, srcarr, dstarr, i, j, k, dx, ncomp);
            }

    // y = 0, z = 0
    EXPECT_NEAR(dstarr(0, 0, 0, 0), 1, 1e-12);
    EXPECT_NEAR(dstarr(1, 0, 0, 0), 1.25, 1e-12);
    EXPECT_NEAR(dstarr(2, 0, 0, 0), 1.5, 1e-12);
    EXPECT_NEAR(dstarr(3, 0, 0, 0), 1.75, 1e-12);
    EXPECT_NEAR(dstarr(4, 0, 0, 0), 2, 1e-12);

    // x = 0, z = 0
    EXPECT_NEAR(dstarr(0, 0, 0, 0), 1, 1e-12);
    EXPECT_NEAR(dstarr(0, 1, 0, 0), 1.5, 1e-12);
    EXPECT_NEAR(dstarr(0, 2, 0, 0), 2, 1e-12);
    EXPECT_NEAR(dstarr(0, 3, 0, 0), 2.5, 1e-12);
    EXPECT_NEAR(dstarr(0, 4, 0, 0), 3, 1e-12);

    // y = 1, z = 0
    EXPECT_NEAR(dstarr(0, 4, 0, 0), 3, 1e-12);
    EXPECT_NEAR(dstarr(1, 4, 0, 0), 3.25, 1e-12);
    EXPECT_NEAR(dstarr(2, 4, 0, 0), 3.5, 1e-12);
    EXPECT_NEAR(dstarr(3, 4, 0, 0), 3.75, 1e-12);
    EXPECT_NEAR(dstarr(4, 4, 0, 0), 4, 1e-12);

    // x = 1, z = 0
    EXPECT_NEAR(dstarr(4, 0, 0, 0), 2, 1e-12);
    EXPECT_NEAR(dstarr(4, 1, 0, 0), 2.5, 1e-12);
    EXPECT_NEAR(dstarr(4, 2, 0, 0), 3, 1e-12);
    EXPECT_NEAR(dstarr(4, 3, 0, 0), 3.5, 1e-12);
    EXPECT_NEAR(dstarr(4, 4, 0, 0), 4, 1e-12);

    // x = 0.5, z = 0
    EXPECT_NEAR(dstarr(2, 0, 0, 0), 1.5, 1e-12);
    EXPECT_NEAR(dstarr(2, 1, 0, 0), 2, 1e-12);
    EXPECT_NEAR(dstarr(2, 2, 0, 0), 2.5, 1e-12);
    EXPECT_NEAR(dstarr(2, 3, 0, 0), 3, 1e-12);
    EXPECT_NEAR(dstarr(2, 4, 0, 0), 3.5, 1e-12);

    // y = 0.5, z = 0
    EXPECT_NEAR(dstarr(0, 2, 0, 0), 2, 1e-12);
    EXPECT_NEAR(dstarr(1, 2, 0, 0), 2.25, 1e-12);
    EXPECT_NEAR(dstarr(2, 2, 0, 0), 2.5, 1e-12);
    EXPECT_NEAR(dstarr(3, 2, 0, 0), 2.75, 1e-12);
    EXPECT_NEAR(dstarr(4, 2, 0, 0), 3, 1e-12);

    // y = 0, z = 1
    EXPECT_NEAR(dstarr(0, 0, 4, 0), 5, 1e-12);
    EXPECT_NEAR(dstarr(1, 0, 4, 0), 5.25, 1e-12);
    EXPECT_NEAR(dstarr(2, 0, 4, 0), 5.5, 1e-12);
    EXPECT_NEAR(dstarr(3, 0, 4, 0), 5.75, 1e-12);
    EXPECT_NEAR(dstarr(4, 0, 4, 0), 6, 1e-12);

    // x = 0, y = 0
    EXPECT_NEAR(dstarr(0, 0, 0, 0), 1, 1e-12);
    EXPECT_NEAR(dstarr(0, 0, 1, 0), 2, 1e-12);
    EXPECT_NEAR(dstarr(0, 0, 2, 0), 3, 1e-12);
    EXPECT_NEAR(dstarr(0, 0, 3, 0), 4, 1e-12);
    EXPECT_NEAR(dstarr(0, 0, 4, 0), 5, 1e-12);

    // x = 0, y = 1
    EXPECT_NEAR(dstarr(0, 4, 0, 0), 3, 1e-12);
    EXPECT_NEAR(dstarr(0, 4, 1, 0), 4, 1e-12);
    EXPECT_NEAR(dstarr(0, 4, 2, 0), 5, 1e-12);
    EXPECT_NEAR(dstarr(0, 4, 3, 0), 6, 1e-12);
    EXPECT_NEAR(dstarr(0, 4, 4, 0), 7, 1e-12);

    // x = 0, z = 0.5
    EXPECT_NEAR(dstarr(0, 0, 2, 0), 3, 1e-12);
    EXPECT_NEAR(dstarr(0, 1, 2, 0), 3.5, 1e-12);
    EXPECT_NEAR(dstarr(0, 2, 2, 0), 4, 1e-12);
    EXPECT_NEAR(dstarr(0, 3, 2, 0), 4.5, 1e-12);
    EXPECT_NEAR(dstarr(0, 4, 2, 0), 5, 1e-12);
}
#endif