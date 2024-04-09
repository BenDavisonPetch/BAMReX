#ifndef UnitDomainTest_H_
#define UnitDomainTest_H_

#include "AmrLevelAdv.H"
#include "Euler/Euler.H"

#include <AMReX_IntVect.H>
#include <AMReX_SPACE.H>
#include <gtest/gtest.h>

#include <AMReX_MultiFab.H>

class UnitDomainTest : public testing::Test
{
  protected:
    int                                          ncells;
    amrex::IntVect                               dom_lo, dom_hi;
    amrex::Box                                   domain;
    amrex::BoxArray                              ba;
    amrex::DistributionMapping                   dm;
    amrex::MultiFab                              mf;
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo, prob_hi;
    amrex::RealBox                               real_box;
    amrex::Geometry                              geom;
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx;

    UnitDomainTest()
    {
    }

    void setup(int ncells_)
    {
        ncells = ncells_;
        dom_lo = amrex::IntVect(AMREX_D_DECL(0, 0, 0));
        dom_hi = amrex::IntVect(AMREX_D_DECL(ncells - 1, ncells - 1, ncells - 1));
        domain = amrex::Box(dom_lo, dom_hi);
        ba = amrex::BoxArray(domain);
        dm = amrex::DistributionMapping(ba);
        mf.define(ba, dm, AmrLevelAdv::NUM_STATE, 0);
        prob_lo = { AMREX_D_DECL(0.0, 0.0, 0.0) };
        prob_hi = { AMREX_D_DECL(1.0, 1.0, 1.0) };
        real_box = amrex::RealBox(AMREX_D_DECL(0.0, 0.0, 0.0), AMREX_D_DECL(1.0, 1.0, 1.0));
        geom = amrex::Geometry(domain, &real_box);
        dx = geom.CellSizeArray();
    }

    /**
     * @brief Gives the indices of the multifab corresponding to a given x value
     */
    amrex::IntVect idx_at(amrex::Real x)
    {
        return amrex::IntVect(
            AMREX_D_DECL((int)((x - prob_lo[0]) / dx[0] - 0.5), 0, 0));
    }
};

class ToroTest1 : public UnitDomainTest
{
  protected:
    amrex::Real density_L = 1.0;
    amrex::Real u_L       = 0.0;
    amrex::Real p_L       = 1.0;
    amrex::Real density_R = 0.125;
    amrex::Real u_R       = 0.0;
    amrex::Real p_R       = 0.1;
    amrex::Real time      = 0.25;

    // Expected intermediate states
    amrex::Real exp_p_interm = 0.30313;
    amrex::Real exp_u_interm = 0.92745;
    amrex::Real exp_density_L_interm = 0.42632;
    amrex::Real exp_density_R_interm = 0.26557;

    // Initial primitive states
    const amrex::GpuArray<amrex::Real, EULER_NCOMP> primv_L{
        density_L, AMREX_D_DECL(u_L, 0, 0), p_L
    };

    const amrex::GpuArray<amrex::Real, EULER_NCOMP> primv_R{
        density_R, AMREX_D_DECL(u_R, 0, 0), p_R
    };

    // Expected intermediate states after evolution to time
    const amrex::GpuArray<amrex::Real, EULER_NCOMP> expected_interm_L{
        exp_density_L_interm, AMREX_D_DECL(exp_u_interm, 0, 0), exp_p_interm
    };

    const amrex::GpuArray<amrex::Real, EULER_NCOMP> expected_interm_R{
        exp_density_R_interm, AMREX_D_DECL(exp_u_interm, 0, 0), exp_p_interm
    };


    ToroTest1()
        : UnitDomainTest()
    {
    }
};

class ToroTest2 : public UnitDomainTest
{
  protected:
    amrex::Real density_L = 1.0;
    amrex::Real u_L       = -2.0;
    amrex::Real p_L       = 0.4;
    amrex::Real density_R = 1.0;
    amrex::Real u_R       = 2.0;
    amrex::Real p_R       = 0.4;
    amrex::Real time      = 0.15;

    // Expected intermediate states
    amrex::Real exp_p_interm = 0.00189;
    amrex::Real exp_u_interm = 0.0;
    amrex::Real exp_density_L_interm = 0.02185;
    amrex::Real exp_density_R_interm = 0.02185;

    const amrex::GpuArray<amrex::Real, EULER_NCOMP> primv_L{
        density_L, AMREX_D_DECL(u_L, 0, 0), p_L
    };

    const amrex::GpuArray<amrex::Real, EULER_NCOMP> primv_R{
        density_R, AMREX_D_DECL(u_R, 0, 0), p_R
    };

    // Expected intermediate states after evolution to time
    const amrex::GpuArray<amrex::Real, EULER_NCOMP> expected_interm_L{
        exp_density_L_interm, AMREX_D_DECL(exp_u_interm, 0, 0), exp_p_interm
    };

    const amrex::GpuArray<amrex::Real, EULER_NCOMP> expected_interm_R{
        exp_density_R_interm, AMREX_D_DECL(exp_u_interm, 0, 0), exp_p_interm
    };

    ToroTest2()
        : UnitDomainTest()
    {
    }
};

class ToroTest3 : public UnitDomainTest
{
  protected:
    amrex::Real density_L = 1.0;
    amrex::Real u_L       = 0.0;
    amrex::Real p_L       = 1000.0;
    amrex::Real density_R = 1.0;
    amrex::Real u_R       = 0.0;
    amrex::Real p_R       = 0.01;
    amrex::Real time      = 0.012;

    // Expected intermediate states
    amrex::Real exp_p_interm = 460.894;
    amrex::Real exp_u_interm = 19.5975;
    amrex::Real exp_density_L_interm = 0.57506;
    amrex::Real exp_density_R_interm = 5.99924;

    const amrex::GpuArray<amrex::Real, EULER_NCOMP> primv_L{
        density_L, AMREX_D_DECL(u_L, 0, 0), p_L
    };

    const amrex::GpuArray<amrex::Real, EULER_NCOMP> primv_R{
        density_R, AMREX_D_DECL(u_R, 0, 0), p_R
    };

    // Expected intermediate states after evolution to time
    const amrex::GpuArray<amrex::Real, EULER_NCOMP> expected_interm_L{
        exp_density_L_interm, AMREX_D_DECL(exp_u_interm, 0, 0), exp_p_interm
    };

    const amrex::GpuArray<amrex::Real, EULER_NCOMP> expected_interm_R{
        exp_density_R_interm, AMREX_D_DECL(exp_u_interm, 0, 0), exp_p_interm
    };

    ToroTest3()
        : UnitDomainTest()
    {
    }
};

class ToroTest4 : public UnitDomainTest
{
  protected:
    amrex::Real density_L = 1.0;
    amrex::Real u_L       = 0.0;
    amrex::Real p_L       = 0.01;
    amrex::Real density_R = 1.0;
    amrex::Real u_R       = 0.0;
    amrex::Real p_R       = 100.0;
    amrex::Real time      = 0.035;

    // Expected intermediate states
    amrex::Real exp_p_interm = 46.0950;
    amrex::Real exp_u_interm = -6.19633;
    amrex::Real exp_density_L_interm = 5.99242;
    amrex::Real exp_density_R_interm = 0.57511;

    const amrex::GpuArray<amrex::Real, EULER_NCOMP> primv_L{
        density_L, AMREX_D_DECL(u_L, 0, 0), p_L
    };

    const amrex::GpuArray<amrex::Real, EULER_NCOMP> primv_R{
        density_R, AMREX_D_DECL(u_R, 0, 0), p_R
    };

    // Expected intermediate states after evolution to time
    const amrex::GpuArray<amrex::Real, EULER_NCOMP> expected_interm_L{
        exp_density_L_interm, AMREX_D_DECL(exp_u_interm, 0, 0), exp_p_interm
    };

    const amrex::GpuArray<amrex::Real, EULER_NCOMP> expected_interm_R{
        exp_density_R_interm, AMREX_D_DECL(exp_u_interm, 0, 0), exp_p_interm
    };

    ToroTest4()
        : UnitDomainTest()
    {
    }
};

class ToroTest5 : public UnitDomainTest
{
  protected:
    amrex::Real density_L = 5.99924;
    amrex::Real u_L       = 19.5975;
    amrex::Real p_L       = 460.894;
    amrex::Real density_R = 5.99242;
    amrex::Real u_R       = -6.19633;
    amrex::Real p_R       = 46.0950;
    amrex::Real time      = 0.035;

    // Expected intermediate states
    amrex::Real exp_p_interm = 1691.64;
    amrex::Real exp_u_interm = 8.68975;
    amrex::Real exp_density_L_interm = 14.2823;
    amrex::Real exp_density_R_interm = 31.0426;

    const amrex::GpuArray<amrex::Real, EULER_NCOMP> primv_L{
        density_L, AMREX_D_DECL(u_L, 0, 0), p_L
    };

    const amrex::GpuArray<amrex::Real, EULER_NCOMP> primv_R{
        density_R, AMREX_D_DECL(u_R, 0, 0), p_R
    };

    // Expected intermediate states after evolution to time
    const amrex::GpuArray<amrex::Real, EULER_NCOMP> expected_interm_L{
        exp_density_L_interm, AMREX_D_DECL(exp_u_interm, 0, 0), exp_p_interm
    };

    const amrex::GpuArray<amrex::Real, EULER_NCOMP> expected_interm_R{
        exp_density_R_interm, AMREX_D_DECL(exp_u_interm, 0, 0), exp_p_interm
    };

    ToroTest5()
        : UnitDomainTest()
    {
    }
};

#endif