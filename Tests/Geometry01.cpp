#include "Geometry/MakeSDF.H"
#include <AMReX_ParmParse.H>
#include <gtest/gtest.h>

using namespace amrex;

#if AMREX_SPACEDIM != 2
#error "2D tests only!"
#endif

TEST(MakeSDF, PolygonUnion1)
{
    {
        ParmParse ppls("ls");
        ppls.add("geom_type", (std::string) "polygon_union");
        ppls.add("polygons", 2);
        ppls.add("polygons_have_fluid_inside", false);
    }
    {
        ParmParse pp("polygon_1");
        pp.add("vertices", 4);
        pp.add("fluid_inside", false);
        pp.addarr("vertex_1", std::vector<Real>{ 1.0, 1.0 });
        pp.addarr("vertex_2", std::vector<Real>{ 1.0, 2.0 });
        pp.addarr("vertex_3", std::vector<Real>{ 2.0, 2.0 });
        pp.addarr("vertex_4", std::vector<Real>{ 2.0, 1.0 });
    }
    {
        ParmParse pp("polygon_2");
        pp.add("vertices", 3);
        pp.add("fluid_inside", false);
        pp.add("anticlockwise", true);
        pp.addarr("vertex_1", std::vector<Real>{ 2.0, 2.0 });
        pp.addarr("vertex_2", std::vector<Real>{ 2.0, 1.0 });
        pp.addarr("vertex_3", std::vector<Real>{ 3.0, 1.5 });
    }
    auto sdf = SDF::make_sdf();
    // In out tests
    EXPECT_TRUE(sdf(RealArray{1.5, 1.1}) > 0);
    EXPECT_TRUE(sdf(RealArray{2.5, 1.5}) > 0);
    EXPECT_TRUE(sdf(RealArray{-1, 1.5}) < 0);
    EXPECT_TRUE(sdf(RealArray{3.5, 1.5}) < 0);
    EXPECT_TRUE(sdf(RealArray{1.5, 2.1}) < 0);

    // Distance tests
    EXPECT_NEAR(sdf(RealArray{0.9, 1.1}), -0.1, 1e-12);
    EXPECT_NEAR(sdf(RealArray{0.5, 0.5}), -0.5*sqrt(2), 1e-12);
    EXPECT_NEAR(sdf(RealArray{1.6, 1.1}), 0.1, 1e-12);
    EXPECT_NEAR(sdf(RealArray{4, 1.5}), -1, 1e-12);
    EXPECT_NEAR(sdf(RealArray{1, 2}), 0, 1e-12);
    EXPECT_NEAR(sdf(RealArray{2, 2}), 0, 1e-12);
    EXPECT_NEAR(sdf(RealArray{3, 1.5}), 0, 1e-12);
}