#include "BoxTest.H"
#include "Euler/NComp.H"
#include "IMEX/IMEX_O1.H"
#include "MFIterLoop.H"
#include "UnitDomainTest.H"

#include <gtest/gtest.h>

using namespace amrex;

TEST_F(ToroTest1, RusanovAdv)
{
    setup(4);
    const Real adia = AmrLevelAdv::h_prob_parm->adiabatic;
    ASSERT_EQ(adia, 1.4);
    ASSERT_EQ(AmrLevelAdv::h_prob_parm->epsilon, 1);
    const auto &consv_L = consv_from_primv(primv_L, adia);
    const auto &consv_R = consv_from_primv(primv_R, adia);

    // Define multifabs
    MultiFab                        statein(ba, dm, EULER_NCOMP, 2);
    MultiFab                        stateout(ba, dm, EULER_NCOMP, 0);
    Array<MultiFab, AMREX_SPACEDIM> fluxes;
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        BoxArray ba = mf.boxArray();
        ba.surroundingNodes(d);
        fluxes[d].define(ba, dm, AmrLevelAdv::NUM_STATE, 0);
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
#if AMREX_SPACEDIM == 1
                            if (i < 2)
                                in(i, j, k, n) = consv_L[n];
                            else
                                in(i, j, k, n) = consv_R[n];
#elif AMREX_SPACEDIM == 3
                            if (i < 2 && j < 2 && k < 2)
                                in(i, j, k, n) = consv_L[n];
                            else
                                in(i, j, k, n) = consv_R[n];
#endif
                        });
        }
    }

    // Advance with Rusanov flux
    const Real dt = 0.1;

    MFIter_loop(
        0, geom, statein, stateout, fluxes, dt,
        [&](const MFIter & /*mfi*/, const Real time, const Box &bx,
            GpuArray<Box, AMREX_SPACEDIM> nbx, const FArrayBox &infab,
            FArrayBox &outfab,
            AMREX_D_DECL(FArrayBox & fx, FArrayBox & fy, FArrayBox & fz),
            GpuArray<Real, AMREX_SPACEDIM> dx, const Real dt)
        {
            advance_rusanov_adv(time, bx, nbx, infab, outfab,
                                AMREX_D_DECL(fx, fy, fz), dx, dt);
        });

    // Now let us check
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    {
        AMREX_D_TERM(const Real dtdx = dt / dx[0];
                     , const Real dtdy = dt / dx[1];
                     , const Real dtdz = dt / dx[2];)
        for (MFIter mfi(stateout, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box  &bx  = mfi.tilebox();
            const auto &in  = statein.array(mfi);
            const auto &out = stateout.array(mfi);
            AMREX_D_TERM(const auto &fluxx = fluxes[0].array(mfi);
                         , const auto &fluxy = fluxes[1].array(mfi);
                         , const auto &fluxz = fluxes[2].array(mfi);)
            ParallelFor(
                bx,
                [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    // Check validity of solution
                    for (int n = 0; n < EULER_NCOMP; ++n)
                        EXPECT_EQ(in(i, j, k, n), out(i, j, k, n));
                    // Check fluxes have been calculated correctly
                    for (int n = 0; n < EULER_NCOMP; ++n)
                        EXPECT_NEAR(
                            out(i, j, k, n),
                            in(i, j, k, n)
                                - (AMREX_D_TERM(dtdx
                                                    * (fluxx(i + 1, j, k, n)
                                                       - fluxx(i, j, k, n)),
                                                +dtdy
                                                    * (fluxy(i, j + 1, k, n)
                                                       - fluxy(i, j, k, n)),
                                                +dtdz
                                                    * (fluxz(i, j, k + 1, n)
                                                       - fluxz(i, j, k, n)))),
                            out(i, j, k, n) * 1e-12);
                });
        }
    }
}

TEST_F(ToroTest2, RusanovAdv)
{
    setup(4);
    const Real adia = AmrLevelAdv::h_prob_parm->adiabatic;
    ASSERT_EQ(adia, 1.4);
    ASSERT_EQ(AmrLevelAdv::h_prob_parm->epsilon, 1);
    const auto &consv_L = consv_from_primv(primv_L, adia);
    const auto &consv_R = consv_from_primv(primv_R, adia);

    // Define multifabs
    MultiFab                        statein(ba, dm, EULER_NCOMP, 2);
    MultiFab                        stateout(ba, dm, EULER_NCOMP, 0);
    Array<MultiFab, AMREX_SPACEDIM> fluxes;
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        BoxArray ba = mf.boxArray();
        ba.surroundingNodes(d);
        fluxes[d].define(ba, dm, AmrLevelAdv::NUM_STATE, 0);
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
#if AMREX_SPACEDIM == 1
                            if (i < 2)
                                in(i, j, k, n) = consv_L[n];
                            else
                                in(i, j, k, n) = consv_R[n];
#elif AMREX_SPACEDIM == 3
                            if (i < 2 && j < 2 && k < 2)
                                in(i, j, k, n) = consv_L[n];
                            else
                                in(i, j, k, n) = consv_R[n];
#endif
                        });
        }
    }

    // Advance with Rusanov flux
    const Real dt = 0.1;

    MFIter_loop(
        0, geom, statein, stateout, fluxes, dt,
        [&](const MFIter & /*mfi*/, const Real time, const Box &bx,
            GpuArray<Box, AMREX_SPACEDIM> nbx, const FArrayBox &infab,
            FArrayBox &outfab,
            AMREX_D_DECL(FArrayBox & fx, FArrayBox & fy, FArrayBox & fz),
            GpuArray<Real, AMREX_SPACEDIM> dx, const Real dt)
        {
            advance_rusanov_adv(time, bx, nbx, infab, outfab,
                                AMREX_D_DECL(fx, fy, fz), dx, dt);
        });

    // Now let us check
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    {
        AMREX_D_TERM(const Real dtdx = dt / dx[0];
                     , const Real dtdy = dt / dx[1];
                     , const Real dtdz = dt / dx[2];)
        for (MFIter mfi(stateout, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box  &bx  = mfi.tilebox();
            const auto &in  = statein.array(mfi);
            const auto &out = stateout.array(mfi);
            AMREX_D_TERM(const auto &fluxx = fluxes[0].array(mfi);
                         , const auto &fluxy = fluxes[1].array(mfi);
                         , const auto &fluxz = fluxes[2].array(mfi);)
            // Check flux is correct
            EXPECT_NEAR(fluxx(0, 0, 0, 0), -2, 1e-12);
            EXPECT_NEAR(fluxx(0, 0, 0, 1), 4, 1e-12);
            EXPECT_NEAR(fluxx(0, 0, 0, 1 + AMREX_SPACEDIM), -4, 1e-12);

            EXPECT_NEAR(fluxx(1, 0, 0, 0), -2, 1e-12);
            EXPECT_NEAR(fluxx(1, 0, 0, 1), 4, 1e-12);
            EXPECT_NEAR(fluxx(1, 0, 0, 1 + AMREX_SPACEDIM), -4, 1e-12);

            EXPECT_NEAR(fluxx(2, 0, 0, 0), 0, 1e-12);
            EXPECT_NEAR(fluxx(2, 0, 0, 1), 0, 1e-12);
            EXPECT_NEAR(fluxx(2, 0, 0, 1 + AMREX_SPACEDIM), 0, 1e-12);

            EXPECT_NEAR(fluxx(3, 0, 0, 0), 2, 1e-12);
            EXPECT_NEAR(fluxx(3, 0, 0, 1), 4, 1e-12);
            EXPECT_NEAR(fluxx(3, 0, 0, 1 + AMREX_SPACEDIM), 4, 1e-12);

            EXPECT_NEAR(fluxx(4, 0, 0, 0), 2, 1e-12);
            EXPECT_NEAR(fluxx(4, 0, 0, 1), 4, 1e-12);
            EXPECT_NEAR(fluxx(4, 0, 0, 1 + AMREX_SPACEDIM), 4, 1e-12);

#if AMREX_SPACEDIM == 3
            EXPECT_NEAR(fluxx(0, 0, 0, 2), 0, 1e-12);
            EXPECT_NEAR(fluxx(0, 0, 0, 3), 0, 1e-12);
            EXPECT_NEAR(fluxx(1, 0, 0, 2), 0, 1e-12);
            EXPECT_NEAR(fluxx(1, 0, 0, 3), 0, 1e-12);
            EXPECT_NEAR(fluxx(2, 0, 0, 2), 0, 1e-12);
            EXPECT_NEAR(fluxx(2, 0, 0, 3), 0, 1e-12);
            EXPECT_NEAR(fluxx(3, 0, 0, 2), 0, 1e-12);
            EXPECT_NEAR(fluxx(3, 0, 0, 3), 0, 1e-12);
            EXPECT_NEAR(fluxx(4, 0, 0, 2), 0, 1e-12);
            EXPECT_NEAR(fluxx(4, 0, 0, 3), 0, 1e-12);

            ParallelFor(
                mfi.nodaltilebox(1), EULER_NCOMP,
                [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                { EXPECT_NEAR(fluxy(i, j, k, n), 0, 1e-12); },
                mfi.nodaltilebox(2), EULER_NCOMP,
                [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                { EXPECT_NEAR(fluxz(i, j, k, n), 0, 1e-12); });
#endif

            ParallelFor(
                bx,
                [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    // Check output corresponds to flux
                    for (int n = 0; n < EULER_NCOMP; ++n)
                        EXPECT_NEAR(
                            out(i, j, k, n),
                            in(i, j, k, n)
                                - (AMREX_D_TERM(dtdx
                                                    * (fluxx(i + 1, j, k, n)
                                                       - fluxx(i, j, k, n)),
                                                +dtdy
                                                    * (fluxy(i, j + 1, k, n)
                                                       - fluxy(i, j, k, n)),
                                                +dtdz
                                                    * (fluxz(i, j, k + 1, n)
                                                       - fluxz(i, j, k, n)))),
                            fabs(out(i, j, k, n)) * 1e-12);
                });
        }
    }
}

#if AMREX_SPACEDIM == 3
TEST_F(BoxTest, RusanovAdv)
{
    setup(true);
    const Real epsilon = AmrLevelAdv::h_prob_parm->epsilon;
    const Real adia    = AmrLevelAdv::h_prob_parm->adiabatic;
    ASSERT_EQ(epsilon, 0.5);
    ASSERT_EQ(adia, 1.6);

    // Advance with Rusanov flux
    const Real dt = 0.1;

    MFIter_loop(
        0, geom, statein, stateout, fluxes, dt,
        [&](const MFIter & /*mfi*/, const Real time, const Box &bx,
            GpuArray<Box, AMREX_SPACEDIM> nbx, const FArrayBox &infab,
            FArrayBox &outfab,
            AMREX_D_DECL(FArrayBox & fx, FArrayBox & fy, FArrayBox & fz),
            GpuArray<Real, AMREX_SPACEDIM> dx, const Real dt)
        {
            advance_rusanov_adv(time, bx, nbx, infab, outfab,
                                AMREX_D_DECL(fx, fy, fz), dx, dt);
        });

    // Now let us check
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    {
        AMREX_D_TERM(const Real dtdx = dt / dx[0];
                     , const Real dtdy = dt / dx[1];
                     , const Real dtdz = dt / dx[2];)
        for (MFIter mfi(stateout, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box  &bx  = mfi.tilebox();
            const auto &in  = statein.array(mfi);
            const auto &out = stateout.array(mfi);
            AMREX_D_TERM(const auto &fluxx = fluxes[0].array(mfi);
                         , const auto &fluxy = fluxes[1].array(mfi);
                         , const auto &fluxz = fluxes[2].array(mfi);)

            // Check fluxes are correct
            const GpuArray<Real, EULER_NCOMP> fluxx_LL{ 2, 2, -4, 6, 7 };
            const GpuArray<Real, EULER_NCOMP> fluxx_RR{ -2, 4, 2, -8, -10.5 };
            const GpuArray<Real, EULER_NCOMP> fluxx_LR{ 1, 7, -4, 1, -1 };
            const GpuArray<Real, EULER_NCOMP> fluxy_LL{ -4, -4, 8, -12, -14 };
            const GpuArray<Real, EULER_NCOMP> fluxy_RR{ -1, 2, 1, -4, -5.25 };
            const GpuArray<Real, EULER_NCOMP> fluxy_LR{ -1.5, 3, 1.5, -6,
                                                        -8.875 };
            const GpuArray<Real, EULER_NCOMP> fluxz_LL{ 6, 6, -12, 18, 21 };
            const GpuArray<Real, EULER_NCOMP> fluxz_RR{ 4, -8, -4, 16, 21 };
            const GpuArray<Real, EULER_NCOMP> fluxz_LR{ 7, 7, -14, 21, 22.5 };

            ParallelFor(
                mfi.nodaltilebox(0), EULER_NCOMP,
                [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                {
                    // Print() << "i j k n = " << i << " " << j << " " << k <<
                    // " " << n << std::endl;
                    if (i > 2 || j > 1 || k > 1)
                    {
                        // it's an RR flux
                        EXPECT_NEAR(fluxx(i, j, k, n), fluxx_RR[n], 1e-12);
                    }
                    else if (i == 2)
                    {
                        // LR flux
                        EXPECT_NEAR(fluxx(i, j, k, n), fluxx_LR[n], 1e-12);
                    }
                    else
                    {
                        // LL flux
                        EXPECT_NEAR(fluxx(i, j, k, n), fluxx_LL[n], 1e-12);
                    }
                },
                mfi.nodaltilebox(1), EULER_NCOMP,
                [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                {
                    if (i > 1 || j > 2 || k > 1)
                    {
                        // it's an RR flux
                        EXPECT_NEAR(fluxy(i, j, k, n), fluxy_RR[n], 1e-12);
                    }
                    else if (j == 2)
                    {
                        // LR flux
                        EXPECT_NEAR(fluxy(i, j, k, n), fluxy_LR[n], 1e-12);
                    }
                    else
                    {
                        // LL flux
                        EXPECT_NEAR(fluxy(i, j, k, n), fluxy_LL[n], 1e-12);
                    }
                },
                mfi.nodaltilebox(2), EULER_NCOMP,
                [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                {
                    if (i > 1 || j > 1 || k > 2)
                    {
                        // it's an RR flux
                        EXPECT_NEAR(fluxz(i, j, k, n), fluxz_RR[n], 1e-12);
                    }
                    else if (k == 2)
                    {
                        // LR flux
                        EXPECT_NEAR(fluxz(i, j, k, n), fluxz_LR[n], 1e-12);
                    }
                    else
                    {
                        // LL flux
                        EXPECT_NEAR(fluxz(i, j, k, n), fluxz_LL[n], 1e-12);
                    }
                });

            ParallelFor(
                bx,
                [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    // Check output corresponds to flux
                    // Print() << "i j k = " << i << " " << j << " " << k << " "
                            // << std::endl;
                    for (int n = 0; n < EULER_NCOMP; ++n)
                        EXPECT_NEAR(
                            out(i, j, k, n),
                            in(i, j, k, n)
                                - (AMREX_D_TERM(dtdx
                                                    * (fluxx(i + 1, j, k, n)
                                                       - fluxx(i, j, k, n)),
                                                +dtdy
                                                    * (fluxy(i, j + 1, k, n)
                                                       - fluxy(i, j, k, n)),
                                                +dtdz
                                                    * (fluxz(i, j, k + 1, n)
                                                       - fluxz(i, j, k, n)))),
                            fabs(out(i, j, k, n)) * 1e-12);
                });
        }
    }
}
#endif