#ifndef BoxTest_H_
#define BoxTest_H_

#include "UnitDomainTest.H"

#if AMREX_SPACEDIM >= 2
class BoxTest : public UnitDomainTest
{
  protected:
#if AMREX_SPACEDIM == 3
    const amrex::GpuArray<amrex::Real, EULER_NCOMP> primv_L{ 2, 1, -2, 3,
                                                             0.6 };
    const amrex::GpuArray<amrex::Real, EULER_NCOMP> primv_R{ 1, -2, -1, 4,
                                                             1.2 };

    const amrex::GpuArray<amrex::Real, EULER_NCOMP> consv_L{ 2, 2, -4, 6, 8 };
    const amrex::GpuArray<amrex::Real, EULER_NCOMP> consv_R{ 1, -2, -1, 4,
                                                             7.25 };
#elif AMREX_SPACEDIM == 2
    const amrex::GpuArray<amrex::Real, EULER_NCOMP> primv_L{ 2, 1, -2, 0.6 };
    const amrex::GpuArray<amrex::Real, EULER_NCOMP> primv_R{ 1, -2, -1, 1.2 };

    const amrex::GpuArray<amrex::Real, EULER_NCOMP> consv_L{ 2, 2, -4, 8 };
    const amrex::GpuArray<amrex::Real, EULER_NCOMP> consv_R{ 1, -2, -1, 7.25 };
#endif

    amrex::MultiFab                               statein, stateout;
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> fluxes;

    BoxTest()
        : UnitDomainTest()
    {
    }

    void setup(bool split_grid)
    {
        using namespace amrex;
        ParmParse pp("prob");
        pp.add("adiabatic", 1.6);
        pp.add("epsilon", 0.5);
        if (split_grid)
            UnitDomainTest::setup(4, 2);
        else
            UnitDomainTest::setup(4);

        statein.define(ba, dm, EULER_NCOMP, 2);
        stateout.define(ba, dm, EULER_NCOMP, 0);

        for (int d = 0; d < AMREX_SPACEDIM; ++d)
        {
            BoxArray ba = stateout.boxArray();
            ba.surroundingNodes(d);
            fluxes[d].define(ba, dm, EULER_NCOMP, 0);
        }

        // Fill input multifab
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        {
            for (MFIter mfi(statein, TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                const Box  &bx = mfi.growntilebox();
                const auto &in = statein.array(mfi);
                ParallelFor(bx, EULER_NCOMP,
                            [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                            {
#if AMREX_SPACEDIM == 3
                                if (i < 2 && j < 2 && k < 2)
                                    in(i, j, k, n) = consv_L[n];
                                else
                                    in(i, j, k, n) = consv_R[n];
#elif AMREX_SPACEDIM == 2
                                if (i < 2 && j < 2)
                                    in(i, j, k, n) = consv_L[n];
                                else
                                    in(i, j, k, n) = consv_R[n];
#endif
                            });
            }
        }
    }
};
#endif

#endif