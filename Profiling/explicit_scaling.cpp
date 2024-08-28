/**
 * Used to determine the scaling properties of MUSCL-Hancock
 * and the MUSCL-Rusanov flux when used on their own
 */
#include <AMReX_BCUtil.H>
#include <AMReX_BC_TYPES.H>
#include <AMReX_PROB_AMR_F.H>
#include <AMReX_Random.H>

#include "Fluxes/Fluxes.H"
#include "IMEX/IMEX_Fluxes.H"
#include "MFIterLoop.H"

#define GRID_SIZE     1024
#define MAX_GRID_SIZE 128
#define NSTEPS        100

#define VERBOSE 0

using namespace amrex;

class Tester
{
  public:
    Tester()
        : dom_lo(AMREX_D_DECL(0, 0, 0))
        , dom_hi(AMREX_D_DECL(GRID_SIZE - 1, GRID_SIZE - 1, GRID_SIZE - 1))
        , domain(dom_lo, dom_hi)
        , ba(domain)
        , prob_lo({ AMREX_D_DECL(0.0, 0.0, 0.0) })
        , prob_hi({ AMREX_D_DECL(1.0, 1.0, 1.0) })
        , real_box(AMREX_D_DECL(0.0, 0.0, 0.0), AMREX_D_DECL(1.0, 1.0, 1.0))
        , geom(domain, &real_box)
        , bc(EULER_NCOMP)
        , dt(0.4 * geom.CellSize(0) / (sqrt(AMREX_SPACEDIM) + 1.6733))
    {
        ba.maxSize(MAX_GRID_SIZE);
        dm = DistributionMapping(ba);

        statein.define(ba, dm, EULER_NCOMP, 2);
        state_toupdate.define(ba, dm, EULER_NCOMP, 2);
        stateout.define(ba, dm, EULER_NCOMP, 1);

        for (int j = 0; j < AMREX_SPACEDIM; j++)
        {
            BoxArray ba = stateout.boxArray();
            ba.surroundingNodes(j);
            fluxes[j].define(ba, dm, EULER_NCOMP, 0);
        }

        for (int n = 0; n < EULER_NCOMP; ++n)
        {
            bc[n] = BCRec(AMREX_D_DECL(BCType::foextrap, BCType::foextrap,
                                       BCType::foextrap),
                          AMREX_D_DECL(BCType::foextrap, BCType::foextrap,
                                       BCType::foextrap));
        }
    }

    void test()
    {
        reset();
        run_mhm();
        reset();
        run_rusanov();
    }

  protected:
    IntVect             dom_lo, dom_hi;
    Box                 domain;
    BoxArray            ba;
    DistributionMapping dm;

    GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo, prob_hi;
    RealBox                               real_box;
    Geometry                              geom;

    Vector<BCRec> bc;

    Real dt;

    MultiFab                        statein;
    MultiFab                        state_toupdate;
    MultiFab                        stateout;
    Array<MultiFab, AMREX_SPACEDIM> fluxes;

    void reset()
    {
        stateout.setVal(0);
        randomize(state_toupdate);
        randomize(statein);
        for (int j = 0; j < AMREX_SPACEDIM; j++)
        {
            fluxes[j].setVal(0);
        }
    }

    void run_mhm()
    {
        for (int iter = 0; iter < NSTEPS; ++iter)
        {
            if (VERBOSE)
                Print() << "[MHM] " << iter << "/" << NSTEPS << std::endl;
            for (int d = 0; d < amrex::SpaceDim; d++)
            {
                const int iOffset = (d == 0 ? 1 : 0);
                const int jOffset = (d == 1 ? 1 : 0);
                const int kOffset = (d == 2 ? 1 : 0);

                // The below loop basically does an MFIter loop but with lots
                // of hidden boilerplate to make things performance portable.
                // Some isms with the
                // muscl implementation: S_new is never touched, the below will
                // update Sborder in place nodal boxes aren't used
                MFIter_loop(
                    0, geom, stateout, statein, fluxes, dt,
                    [&](const amrex::MFIter & /*mfi*/, const amrex::Real time,
                        const amrex::Box &bx,
                        amrex::GpuArray<amrex::Box, AMREX_SPACEDIM> /*nbx*/,
                        const amrex::FArrayBox & /*statein*/,
                        amrex::FArrayBox &state,
                        AMREX_D_DECL(amrex::FArrayBox & fx,
                                     amrex::FArrayBox & fy,
                                     amrex::FArrayBox & fz),
                        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx,
                        const amrex::Real                            dt)
                    {
                        const auto &arr = state.array();
#if AMREX_SPACEDIM == 1
                        const auto &fluxArr = fx.array();
#elif AMREX_SPACEDIM == 2
                        const auto &fluxArr
                            = (d == 0) ? fx.array() : fy.array();
#else
                        const auto &fluxArr
                            = (d == 0) ? fx.array()
                                       : ((d == 1) ? fy.array() : fz.array());
#endif

                        switch (d)
                        {
                        case 0:
                            compute_MUSCL_hancock_flux<0>(time, bx, fluxArr,
                                                          arr, dx, dt);
                            break;
#if AMREX_SPACEDIM >= 2
                        case 1:
                            compute_MUSCL_hancock_flux<1>(time, bx, fluxArr,
                                                          arr, dx, dt);
                            break;
#if AMREX_SPACEDIM == 3
                        case 2:
                            compute_MUSCL_hancock_flux<2>(time, bx, fluxArr,
                                                          arr, dx, dt);
                            break;
#endif
#endif
                        }

                        // Conservative update
                        ParallelFor(bx, EULER_NCOMP,
                                    [=] AMREX_GPU_DEVICE(int i, int j, int k,
                                                         int n) noexcept
                                    {
                                        // Conservative update formula
                                        arr(i, j, k, n)
                                            = arr(i, j, k, n)
                                              - (dt / dx[d])
                                                    * (fluxArr(i + iOffset,
                                                               j + jOffset,
                                                               k + kOffset, n)
                                                       - fluxArr(i, j, k, n));
                                    });
                    },
                    0, { d });

                // We need to compute boundary conditions again after each
                // update
                fill_boundary(statein);
            }
            MultiFab::Copy(stateout, statein, 0, 0, EULER_NCOMP, 0);
        }
    }

    void run_rusanov()
    {
        for (int iter = 0; iter < NSTEPS; ++iter)
        {
            if (VERBOSE)
                Print() << "[RUS] " << iter << "/" << NSTEPS << std::endl;
            const auto dx = geom.CellSizeArray();
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
            {
                for (MFIter mfi(stateout, TilingIfNotGPU()); mfi.isValid();
                     ++mfi)
                {
                    const Box                 &bx = mfi.tilebox();
                    GpuArray<Box, BL_SPACEDIM> nbx;
                    AMREX_D_TERM(nbx[0] = mfi.nodaltilebox(0);
                                 , nbx[1] = mfi.nodaltilebox(1);
                                 , nbx[2] = mfi.nodaltilebox(2));
                    advance_MUSCL_rusanov_adv(
                        0, bx, nbx, statein[mfi], state_toupdate[mfi],
                        stateout[mfi],
                        AMREX_D_DECL(fluxes[0][mfi], fluxes[1][mfi],
                                     fluxes[2][mfi]),
                        dx, dt);
                }
            }

            MultiFab::Copy(statein, stateout, 0, 0, EULER_NCOMP, 0);
            MultiFab::Copy(state_toupdate, stateout, 0, 0, EULER_NCOMP, 0);
            fill_boundary(statein);
            fill_boundary(state_toupdate);
        }
    }

    void randomize(MultiFab &mf)
    {
        // randomizes state with
        // density = [1,2]
        // pressure = [1,2]
        // u,v,w = [-1, 1]
        // so max wave speed is sqrt(SPACEDIM) + 1.67332
        const Real adia = AmrLevelAdv::h_parm->adiabatic;
        const Real eps  = AmrLevelAdv::h_parm->epsilon;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        {
            for (MFIter mfi(mf, TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                const auto &bx  = mfi.tilebox();
                const auto &arr = mf.array(mfi);
                ParallelFor(bx,
                            [=] AMREX_GPU_DEVICE(int i, int j, int k)
                            {
                                GpuArray<Real, EULER_NCOMP> primv;
                                FillRandom(primv.begin(), EULER_NCOMP);
                                primv[0] += 1; // density in range [1,2]
                                primv[1 + AMREX_SPACEDIM]
                                    += 1; // p in range [1,2]
                                for (int n = 1; n < AMREX_SPACEDIM + 1; ++n)
                                {
                                    primv[n] -= 0.5;
                                    primv[n] *= 2; // u,v,w in range [-1,1]
                                }
                                const auto consv
                                    = consv_from_primv(primv, adia, eps);
                                for (int n = 0; n < EULER_NCOMP; ++n)
                                    arr(i, j, k, n) = consv[n];
                            });
            }
        }

        fill_boundary(mf);
    }

    void fill_boundary(MultiFab &mf)
    {
        mf.FillBoundary(geom.periodicity());
        FillDomainBoundary(mf, geom, bc);
    }
};

amrex::LevelBld *getLevelBld();

int main(int argc, char *argv[])
{
    Initialize(argc, argv);

    ParmParse pp("prob");
    pp.add("adiabatic", 1.4);

    auto adv_bld = getLevelBld();
    adv_bld->variableSetUp();

    amrex::InitRandom(39672806746);

    Tester tester;

    tester.test();

    adv_bld->variableCleanUp();
    Finalize();
}