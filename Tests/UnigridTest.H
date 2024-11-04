#ifndef UnigridTest_H_
#define UnigridTest_H_
/**
 * An updated version of UnitDomainTest used for newer tests including MHD that
 * bypasses AmrLevelAdv
 */
#include <gtest/gtest.h>

#include "System/PlasmaSystem.H"
#include "Multidim.H"

using namespace amrex;

class UnigridTest : public testing::Test
{
  protected:
    Box                 domain;
    BoxArray            ba;
    DistributionMapping dm;
    RealBox             real_box;
    Geometry            geom;
    System             *system;
    Parm               *h_parm;
    Parm               *d_parm;

    UnigridTest() {}

    void setup(const IntVect &dom_lo, const IntVect &dom_hi,
               const GpuArray<Real, AMREX_SPACEDIM> &prob_lo,
               const GpuArray<Real, AMREX_SPACEDIM> &prob_hi, ParmParse &pp,
               ParmParse &pplimiting, int max_grid_size = -1)
    {
        domain = Box(dom_lo, dom_hi);
        ba     = BoxArray(domain);
        if (max_grid_size > 0)
            ba.maxSize(max_grid_size);
        dm       = DistributionMapping(ba);
        real_box = RealBox(prob_lo.data(), prob_hi.data());
        geom     = Geometry(domain, &real_box);
        system   = System::NewSystem(pp);

        h_parm = new Parm{};
        d_parm = (Parm *)The_Arena()->alloc(sizeof(Parm));
        system->ReadParameters(*h_parm, pp, pplimiting);
        Gpu::copy(Gpu::hostToDevice, h_parm, h_parm + 1, d_parm);
    }

    void TearDown() override
    {
        if (system)
        {
            delete system;
            system = nullptr;
        }
        if (h_parm)
        {
            delete h_parm;
            h_parm = nullptr;
        }
        if (d_parm)
        {
            The_Arena()->free(d_parm);
            d_parm = nullptr;
        }
    }
};

#endif