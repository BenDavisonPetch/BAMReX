#ifndef hllc_test_H_
#define hllc_test_H_

/**
 * This file contains test fixtures for HLLC wave speed estimates
 */

#include "AmrLevelAdv.H"
#include "System/Euler/Euler.H"

#include <AMReX_Arena.H>
#include <gtest/gtest.h>

using namespace amrex;

class HLLCTest : public testing::Test
{
  protected:
    const int                                    ncells = 2;
    amrex::IntVect                               dom_lo, dom_hi;
    amrex::Box                                   domain;
    amrex::BoxArray                              ba;
    amrex::DistributionMapping                   dm;
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo, prob_hi;
    amrex::RealBox                               real_box;
    amrex::Geometry                              geom;
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx;
    amrex::FArrayBox                             Lfab, Rfab;
    amrex::Array4<amrex::Real>                   dataL, dataR;

    virtual ~HLLCTest() = 0;

    HLLCTest()
        : dom_lo{ AMREX_D_DECL(0, 0, 0) }
        , dom_hi{ AMREX_D_DECL(ncells, ncells, ncells) }
        , domain(dom_lo, dom_hi)
        , ba(domain)
        , dm(ba)
        , prob_lo({ AMREX_D_DECL(0.0, 0.0, 0.0) })
        , prob_hi({ AMREX_D_DECL(1.0, 1.0, 1.0) })
        , real_box(AMREX_D_DECL(0.0, 0.0, 0.0), AMREX_D_DECL(1.0, 1.0, 1.0))
        , geom(domain, &real_box)
        , dx(geom.CellSizeArray())
        , Lfab(domain, EULER_NCOMP, amrex::The_Async_Arena())
        , Rfab(domain, EULER_NCOMP, amrex::The_Async_Arena())
        , dataL(Lfab.array())
        , dataR(Rfab.array())
    {
    }

    void init(const amrex::GpuArray<amrex::Real, EULER_NCOMP> &primv_L,
              const amrex::GpuArray<amrex::Real, EULER_NCOMP> &primv_R,
              amrex::Real adiabatic, amrex::Real epsilon)
    {
        const auto consv_L = consv_from_primv(primv_L, adiabatic, epsilon);
        const auto consv_R = consv_from_primv(primv_R, adiabatic, epsilon);

        ParallelFor(domain, EULER_NCOMP,
                    [=](int i, int j, int k, int n)
                    {
                        dataL(i, j, k, n) = consv_L[n];
                        dataR(i, j, k, n) = consv_R[n];
                    });
    }

    /**
     * @brief Gives the indices of the array4 corresponding to a given value of
     * x
     */
    amrex::IntVect idx_at(amrex::Real x)
    {
        return amrex::IntVect(
            AMREX_D_DECL((int)((x - prob_lo[0]) / dx[0] - 0.5), 0, 0));
    }
};

inline HLLCTest::~HLLCTest() {}

// rar-shock
class Toro2020Test1 : public HLLCTest
{
  protected:
    Real density_L = 1.0;
    Real u_L       = 0.0;
    Real p_L       = 1.0;
    Real density_R = 1.0;
    Real u_R       = 0.0;
    Real p_R       = 0.1;

    // Expected intermediate states
    Real exp_p_interm = 0.5219;
    Real exp_u_interm = 0.5248;

    // Expected wave speeds
    Real exp_exact_S_R = 0.8039;
    Real exp_TMSb_S_R  = 0.8080;
    Real exp_To_S_R    = 0.8134;
    Real exp_exact_S_L = -1.8132;
    Real exp_TMSb_S_L  = -1.1832;
    Real exp_To_S_L    = -1.1832;

    Real adiabatic = 1.4;

    Real exp_TMSb_S_interm
        = (p_R - p_L + density_L * u_L * (exp_TMSb_S_L - u_L)
           - density_R * u_R * (exp_TMSb_S_R - u_R))
          / (density_L * (exp_TMSb_S_L - u_L)
             - density_R * (exp_TMSb_S_R - u_R));

    Real exp_To_S_interm
        = (p_R - p_L + density_L * u_L * (exp_To_S_L - u_L)
           - density_R * u_R * (exp_To_S_R - u_R))
          / (density_L * (exp_To_S_L - u_L) - density_R * (exp_To_S_R - u_R));

    Real exp_exact_S_interm
        = (p_R - p_L + density_L * u_L * (exp_exact_S_L - u_L)
           - density_R * u_R * (exp_exact_S_R - u_R))
          / (density_L * (exp_exact_S_L - u_L)
             - density_R * (exp_exact_S_R - u_R));

    // Initial primitive states
    const GpuArray<Real, EULER_NCOMP> primv_L{ density_L,
                                               AMREX_D_DECL(u_L, 0, 0), p_L };

    const GpuArray<Real, EULER_NCOMP> primv_R{ density_R,
                                               AMREX_D_DECL(u_R, 0, 0), p_R };

    ~Toro2020Test1() override = default;

    Toro2020Test1()
        : HLLCTest()
    {
        init(primv_L, primv_R, adiabatic, 1);
    }
};

// rar-shock
class Toro2020Test2 : public HLLCTest
{
  protected:
    Real density_L = 1.0;
    Real u_L       = 0.0;
    Real p_L       = 1.0;
    Real density_R = 0.125;
    Real u_R       = 0.0;
    Real p_R       = 0.1;

    // Expected intermediate states
    Real exp_p_interm = 0.3031;
    Real exp_u_interm = 0.9274;

    // Expected wave speeds
    Real exp_exact_S_R = 1.7522;
    Real exp_TMSb_S_R  = 1.7554;
    Real exp_To_S_R    = 1.7621;
    Real exp_exact_S_L = -1.1832;
    Real exp_TMSb_S_L  = -1.1832;
    Real exp_To_S_L    = -1.1832;

    Real adiabatic = 1.4;

    Real exp_TMSb_S_interm
        = (p_R - p_L + density_L * u_L * (exp_TMSb_S_L - u_L)
           - density_R * u_R * (exp_TMSb_S_R - u_R))
          / (density_L * (exp_TMSb_S_L - u_L)
             - density_R * (exp_TMSb_S_R - u_R));

    Real exp_To_S_interm
        = (p_R - p_L + density_L * u_L * (exp_To_S_L - u_L)
           - density_R * u_R * (exp_To_S_R - u_R))
          / (density_L * (exp_To_S_L - u_L) - density_R * (exp_To_S_R - u_R));

    Real exp_exact_S_interm
        = (p_R - p_L + density_L * u_L * (exp_exact_S_L - u_L)
           - density_R * u_R * (exp_exact_S_R - u_R))
          / (density_L * (exp_exact_S_L - u_L)
             - density_R * (exp_exact_S_R - u_R));

    // Initial primitive states
    const GpuArray<Real, EULER_NCOMP> primv_L{ density_L,
                                               AMREX_D_DECL(u_L, 0, 0), p_L };

    const GpuArray<Real, EULER_NCOMP> primv_R{ density_R,
                                               AMREX_D_DECL(u_R, 0, 0), p_R };

    ~Toro2020Test2() override = default;

    Toro2020Test2()
        : HLLCTest()
    {
        init(primv_L, primv_R, adiabatic, 1);
    }
};

// shock-rar
class Toro2020Test4 : public HLLCTest
{
  protected:
    Real density_L = 1.0;
    Real u_L       = 0.0;
    Real p_L       = 0.01;
    Real density_R = 1.0;
    Real u_R       = 0.0;
    Real p_R       = 1000.0;

    // Expected intermediate states
    Real exp_p_interm = 460.8938;
    Real exp_u_interm = -19.5975;

    // Expected wave speeds
    Real exp_exact_S_R = 37.4166;
    Real exp_TMSb_S_R  = 37.4166;
    Real exp_To_S_R    = 37.4166;
    Real exp_exact_S_L = -23.5175;
    Real exp_TMSb_S_L  = -33.0886;
    Real exp_To_S_L    = -33.0899;

    Real adiabatic = 1.4;

    Real exp_TMSb_S_interm
        = (p_R - p_L + density_L * u_L * (exp_TMSb_S_L - u_L)
           - density_R * u_R * (exp_TMSb_S_R - u_R))
          / (density_L * (exp_TMSb_S_L - u_L)
             - density_R * (exp_TMSb_S_R - u_R));

    Real exp_To_S_interm
        = (p_R - p_L + density_L * u_L * (exp_To_S_L - u_L)
           - density_R * u_R * (exp_To_S_R - u_R))
          / (density_L * (exp_To_S_L - u_L) - density_R * (exp_To_S_R - u_R));

    Real exp_exact_S_interm
        = (p_R - p_L + density_L * u_L * (exp_exact_S_L - u_L)
           - density_R * u_R * (exp_exact_S_R - u_R))
          / (density_L * (exp_exact_S_L - u_L)
             - density_R * (exp_exact_S_R - u_R));

    // Initial primitive states
    const GpuArray<Real, EULER_NCOMP> primv_L{ density_L,
                                               AMREX_D_DECL(u_L, 0, 0), p_L };

    const GpuArray<Real, EULER_NCOMP> primv_R{ density_R,
                                               AMREX_D_DECL(u_R, 0, 0), p_R };

    ~Toro2020Test4() override = default;

    Toro2020Test4()
        : HLLCTest()
    {
        init(primv_L, primv_R, adiabatic, 1);
    }
};

// shock-shock
class Toro2020Test5 : public HLLCTest
{
  protected:
    Real density_L = 6.0;
    Real u_L       = 8.0;
    Real p_L       = 460.0;
    Real density_R = 6.0;
    Real u_R       = -6.0;
    Real p_R       = 46.0;

    // Expected intermediate states
    Real exp_p_interm = 790.2928;
    Real exp_u_interm = 3.8194;

    // Expected wave speeds
    Real exp_exact_S_R = 6.6330;
    Real exp_TMSb_S_R  = 7.1170;
    Real exp_To_S_R    = 7.5400;
    Real exp_exact_S_L = -5.1678;
    Real exp_TMSb_S_L  = -5.6329;
    Real exp_To_S_L    = -6.0404;

    Real adiabatic = 1.4;

    Real exp_TMSb_S_interm
        = (p_R - p_L + density_L * u_L * (exp_TMSb_S_L - u_L)
           - density_R * u_R * (exp_TMSb_S_R - u_R))
          / (density_L * (exp_TMSb_S_L - u_L)
             - density_R * (exp_TMSb_S_R - u_R));

    Real exp_To_S_interm
        = (p_R - p_L + density_L * u_L * (exp_To_S_L - u_L)
           - density_R * u_R * (exp_To_S_R - u_R))
          / (density_L * (exp_To_S_L - u_L) - density_R * (exp_To_S_R - u_R));

    Real exp_exact_S_interm
        = (p_R - p_L + density_L * u_L * (exp_exact_S_L - u_L)
           - density_R * u_R * (exp_exact_S_R - u_R))
          / (density_L * (exp_exact_S_L - u_L)
             - density_R * (exp_exact_S_R - u_R));

    // Initial primitive states
    const GpuArray<Real, EULER_NCOMP> primv_L{ density_L,
                                               AMREX_D_DECL(u_L, 0, 0), p_L };

    const GpuArray<Real, EULER_NCOMP> primv_R{ density_R,
                                               AMREX_D_DECL(u_R, 0, 0), p_R };

    ~Toro2020Test5() override = default;

    Toro2020Test5()
        : HLLCTest()
    {
        init(primv_L, primv_R, adiabatic, 1);
    }
};

// rar-rar
class Toro2020Test7 : public HLLCTest
{
  protected:
    Real density_L = 1.0;
    Real u_L       = -2.0;
    Real p_L       = 0.4;
    Real density_R = 1.0;
    Real u_R       = 2.0;
    Real p_R       = 0.4;

    // Expected intermediate states
    Real exp_p_interm = 0.0019;
    Real exp_u_interm = 0.0000;

    // Expected wave speeds
    Real exp_exact_S_R = 2.7483;
    Real exp_TMSb_S_R  = 2.7483;
    Real exp_To_S_R    = 2.7483;
    Real exp_exact_S_L = -2.7483;
    Real exp_TMSb_S_L  = -2.7483;
    Real exp_To_S_L    = -2.7483;

    Real adiabatic = 1.4;

    Real exp_TMSb_S_interm
        = (p_R - p_L + density_L * u_L * (exp_TMSb_S_L - u_L)
           - density_R * u_R * (exp_TMSb_S_R - u_R))
          / (density_L * (exp_TMSb_S_L - u_L)
             - density_R * (exp_TMSb_S_R - u_R));

    Real exp_To_S_interm
        = (p_R - p_L + density_L * u_L * (exp_To_S_L - u_L)
           - density_R * u_R * (exp_To_S_R - u_R))
          / (density_L * (exp_To_S_L - u_L) - density_R * (exp_To_S_R - u_R));

    Real exp_exact_S_interm
        = (p_R - p_L + density_L * u_L * (exp_exact_S_L - u_L)
           - density_R * u_R * (exp_exact_S_R - u_R))
          / (density_L * (exp_exact_S_L - u_L)
             - density_R * (exp_exact_S_R - u_R));

    // Initial primitive states
    const GpuArray<Real, EULER_NCOMP> primv_L{ density_L,
                                               AMREX_D_DECL(u_L, 0, 0), p_L };

    const GpuArray<Real, EULER_NCOMP> primv_R{ density_R,
                                               AMREX_D_DECL(u_R, 0, 0), p_R };

    ~Toro2020Test7() override = default;

    Toro2020Test7()
        : HLLCTest()
    {
        init(primv_L, primv_R, adiabatic, 1);
    }
};

// shock-shock
class Toro2020Test5Scaled : public HLLCTest
{
  protected:
    Real density_0 = 6.0;
    Real u_0       = 7;
    Real p_0       = 46;
    Real x_0       = 1;
    Real t_0       = x_0 / u_0;
    Real epsilon   = density_0 * u_0 * u_0 / p_0;

    Real density_L = 6.0;
    Real u_L       = 8.0;
    Real p_L       = 460.0;
    Real density_R = 6.0;
    Real u_R       = -6.0;
    Real p_R       = 46.0;

    // Expected intermediate states
    Real exp_p_interm = 790.2928;
    Real exp_u_interm = 3.8194;

    // Expected wave speeds
    Real exp_exact_S_R = 6.6330;
    Real exp_TMSb_S_R  = 7.1170;
    Real exp_To_S_R    = 7.5400;
    Real exp_exact_S_L = -5.1678;
    Real exp_TMSb_S_L  = -5.6329;
    Real exp_To_S_L    = -6.0404;

    Real adiabatic = 1.4;

    Real exp_TMSb_S_interm
        = (p_R - p_L + density_L * u_L * (exp_TMSb_S_L - u_L)
           - density_R * u_R * (exp_TMSb_S_R - u_R))
          / (density_L * (exp_TMSb_S_L - u_L)
             - density_R * (exp_TMSb_S_R - u_R));

    Real exp_To_S_interm
        = (p_R - p_L + density_L * u_L * (exp_To_S_L - u_L)
           - density_R * u_R * (exp_To_S_R - u_R))
          / (density_L * (exp_To_S_L - u_L) - density_R * (exp_To_S_R - u_R));

    Real exp_exact_S_interm
        = (p_R - p_L + density_L * u_L * (exp_exact_S_L - u_L)
           - density_R * u_R * (exp_exact_S_R - u_R))
          / (density_L * (exp_exact_S_L - u_L)
             - density_R * (exp_exact_S_R - u_R));

    // Initial primitive states
    const GpuArray<Real, EULER_NCOMP> primv_L{ density_L / density_0,
                                               AMREX_D_DECL(u_L / u_0, 0, 0),
                                               p_L / p_0 };

    const GpuArray<Real, EULER_NCOMP> primv_R{ density_R / density_0,
                                               AMREX_D_DECL(u_R / u_0, 0, 0),
                                               p_R / p_0 };

    ~Toro2020Test5Scaled() override = default;

    Toro2020Test5Scaled()
        : HLLCTest()
    {
        init(primv_L, primv_R, adiabatic, epsilon);
    }
};

#endif