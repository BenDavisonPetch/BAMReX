#include <AMReX_Array.H>
#include <AMReX_MLABecLaplacian.H>
#include <AMReX_MLMG.H>
#include <gtest/gtest.h>

#include <AMReX_MultiFab.H>

#include "LinearSolver/MLALaplacianGrad.H"
#include "UnitDomainTest.H"

using namespace amrex;

class MLABecLapGradTest : public UnitDomainTest
{
  protected:
    const amrex::BCRec tbc = amrex::BCRec(
        AMREX_D_DECL(amrex::BCType::foextrap, amrex::BCType::foextrap,
                     amrex::BCType::foextrap),
        AMREX_D_DECL(amrex::BCType::foextrap, amrex::BCType::foextrap,
                     amrex::BCType::foextrap));
    const amrex::Vector<amrex::BCRec> bcs{ tbc, AMREX_D_DECL(tbc, tbc, tbc),
                                           tbc };

    Real a;
    Real b;
    Real c;
    Real rhs_norm;

    amrex::MultiFab stateout, alpha, gamma, beta_cc, source;
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> beta;
    // amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> fluxes;

    MLABecLaplacianGrad lapgrad;

    MLABecLapGradTest()
        : UnitDomainTest()
    {
    }

    void setup(bool split_grid, Real a_, Real b_, Real c_)
    {
        using namespace amrex;
        a = a_;
        b = b_;
        c = c_;

        ParmParse pp("prob");
        pp.add("adiabatic", 1.6);
        pp.add("epsilon", 0.5);
        if (split_grid)
            UnitDomainTest::setup(8, 4);
        else
            UnitDomainTest::setup(8);

        beta_cc.define(ba, dm, 1, 1);
        alpha.define(ba, dm, 1, 0);
        gamma.define(ba, dm, AMREX_SPACEDIM, 0);
        stateout.define(ba, dm, 1, 0);
        source.define(ba, dm, 1, 0);

        for (int d = 0; d < AMREX_SPACEDIM; ++d)
        {
            BoxArray ba = stateout.boxArray();
            ba.surroundingNodes(d);
            // fluxes[d].define(ba, dm, EULER_NCOMP, 0);
            beta[d].define(ba, dm, 1, 0);
        }

        // Fill input multifabs
        stateout.setVal(0);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        {
            for (MFIter mfi(stateout, TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                const Box  &bx          = mfi.tilebox();
                const Box  &gbx         = mfi.growntilebox(1);
                const auto &alpha_arr   = alpha.array(mfi);
                const auto &beta_cc_arr = beta_cc.array(mfi);
                const auto &gamma_arr   = gamma.array(mfi);
                const auto &src         = source.array(mfi);
                ParallelFor(
                    bx,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k)
                    {
                        AMREX_D_TERM(
                            const Real x = (i + 0.5) * dx[0] + prob_lo[0];
                            , const Real y = (j + 0.5) * dx[1] + prob_lo[1];
                            , const Real z = (k + 0.5) * dx[2] + prob_lo[2];)
                        alpha_arr(i, j, k) = AMREX_D_TERM(
                            amrex::Math::powi<2>(amrex::Math::sinpi(2 * x)),
                            *amrex::Math::powi<2>(amrex::Math::cospi(4 * y)),
                            *amrex::Math::powi<2>(amrex::Math::sinpi(6 * z)));
                        src(i, j, k) = AMREX_D_TERM(
                            amrex::Math::powi<3>(amrex::Math::sinpi(2 * x)),
                            *amrex::Math::cospi(4 * y),
                            *(amrex::Math::powi<2>(amrex::Math::cospi(2 * z)))
                                + 0.2);
                        gamma_arr(i, j, k, 0) = AMREX_D_TERM(
                            amrex::Math::powi<4>(amrex::Math::cospi(x)),
                            *amrex::Math::powi<2>(amrex::Math::cospi(2 * y)),
                            *amrex::Math::powi<2>(amrex::Math::cospi(3 * z)));
#if AMREX_SPACEDIM >= 2
                        gamma_arr(i, j, k, 1) = AMREX_D_TERM(
                            amrex::Math::powi<2>(amrex::Math::cospi(3 * x)),
                            *amrex::Math::powi<4>(amrex::Math::sinpi(2 * y)),
                            *amrex::Math::powi<4>(amrex::Math::cospi(z)));
#if AMREX_SPACEDIM >= 3
                        gamma_arr(i, j, k, 2) = AMREX_D_TERM(
                            amrex::Math::powi<2>(amrex::Math::sinpi(3 * x)
                                                 + 1),
                            *amrex::Math::powi<2>(amrex::Math::cospi(3 * y)),
                            *amrex::Math::powi<4>(amrex::Math::sinpi(2 * z)));
#endif
#endif
                    });
                ParallelFor(
                    gbx,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k)
                    {
                        AMREX_D_TERM(
                            const Real x = (i + 0.5) * dx[0] + prob_lo[0];
                            , const Real y = (j + 0.5) * dx[1] + prob_lo[1];
                            , const Real z = (k + 0.5) * dx[2] + prob_lo[2];)
                        beta_cc_arr(i, j, k) = AMREX_D_TERM(
                            amrex::Math::sinpi(2 * x) + 2,
                            +0.5 * amrex::Math::cospi(4 * y),
                            +amrex::Math::powi<2>(amrex::Math::cospi(z)));
                    });
            }
        }

        amrex::average_cellcenter_to_face(amrex::GetArrOfPtrs(beta), beta_cc,
                                          geom);
        rhs_norm = source.norm2();

        lapgrad.define({ geom }, { ba }, { dm });

        // BCs
        {
            Array<LinOpBCType, AMREX_SPACEDIM> lobc, hibc;
            for (int d = 0; d < AMREX_SPACEDIM; ++d)
            {
                if (geom.isPeriodic(d))
                {
                    lobc[d] = LinOpBCType::Periodic;
                    hibc[d] = LinOpBCType::Periodic;
                }
                else
                {
                    lobc[d] = LinOpBCType::Neumann;
                    hibc[d] = LinOpBCType::Neumann;
                }
            }
            lapgrad.setDomainBC(lobc, hibc);
        }

        lapgrad.setLevelBC(0, nullptr);

        lapgrad.setScalars(a, b, c);
        lapgrad.setACoeffs(0, alpha);
        lapgrad.setBCoeffs(0, amrex::GetArrOfConstPtrs(beta));
        lapgrad.setCCoeffs(0, gamma);
    }
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE Real residual(
    int i, int j, int k, Real a, Real b, Real c,
    const Array4<const Real> &alpha,
    AMREX_D_DECL(const Array4<const Real> &bx, const Array4<const Real> &by,
                 const Array4<const Real> &bz),
    const Array4<const Real> &gamma, const Array4<const Real> &rhs,
    const Array4<const Real> &soln, const GpuArray<Real, AMREX_SPACEDIM> &dx)
{
    const Real aterm = a * alpha(i, j, k) * soln(i, j, k);
    AMREX_D_TERM(const Real ddx = 1 / (dx[0] * dx[0]);
                 , const Real ddy = 1 / (dx[1] * dx[1]);
                 , const Real ddz = 1 / (dx[2] * dx[2]);)
    AMREX_D_TERM(const Real idx = 1 / dx[0];, const Real idy = 1 / dx[1];
                 , const Real                            idz = 1 / dx[2];)
    const Real bterm
        = -b
          * (AMREX_D_TERM(
              ddx
                  * (bx(i + 1, j, k) * (soln(i + 1, j, k) - soln(i, j, k))
                     - bx(i, j, k) * (soln(i, j, k) - soln(i - 1, j, k))),
              +ddy
                  * (by(i, j + 1, k) * (soln(i, j + 1, k) - soln(i, j, k))
                     - by(i, j, k) * (soln(i, j, k) - soln(i, j - 1, k))),
              +ddz
                  * (bz(i, j, k + 1) * (soln(i, j, k + 1) - soln(i, j, k))
                     - bz(i, j, k) * (soln(i, j, k) - soln(i, j, k - 1)))));
    const Real cterm
        = 0.5 * c
          * (AMREX_D_TERM(idx * gamma(i, j, k, 0)
                              * (soln(i + 1, j, k) - soln(i - 1, j, k)),
                          +idy * gamma(i, j, k, 1)
                              * (soln(i, j + 1, k) - soln(i, j - 1, k)),
                          +idz * gamma(i, j, k, 2)
                              * (soln(i, j, k + 1) - soln(i, j, k - 1))));
    return aterm + bterm + cterm - rhs(i, j, k);
}

TEST_F(MLABecLapGradTest, BuiltInLaplacian)
{
    setup(false, 1, 2, 0);

    MLABecLaplacian mlabeclap({ geom }, { ba }, { dm });

    // BCs
    {
        Array<LinOpBCType, AMREX_SPACEDIM> lobc, hibc;
        for (int d = 0; d < AMREX_SPACEDIM; ++d)
        {
            if (geom.isPeriodic(d))
            {
                lobc[d] = LinOpBCType::Periodic;
                hibc[d] = LinOpBCType::Periodic;
            }
            else
            {
                lobc[d] = LinOpBCType::Neumann;
                hibc[d] = LinOpBCType::Neumann;
            }
        }
        mlabeclap.setDomainBC(lobc, hibc);
    }

    mlabeclap.setLevelBC(0, nullptr);

    mlabeclap.setScalars(a, b);
    mlabeclap.setACoeffs(0, alpha);
    mlabeclap.setBCoeffs(0, amrex::GetArrOfConstPtrs(beta));

    MLMG solver(mlabeclap);
    solver.setMaxIter(100);
    solver.setMaxFmgIter(0);
    solver.setVerbose(1);
    solver.setBottomVerbose(1);

    const Real TOL_RES = 1e-12;
    const Real TOL_ABS = 0;

    solver.solve({ &stateout }, { &source }, TOL_RES, TOL_ABS);

    for (MFIter mfi(stateout); mfi.isValid(); ++mfi)
    {
        // check values without boundary conditions
        const auto &sbx  = amrex::grow(mfi.validbox(), -1);
        const auto &out  = stateout.const_array(mfi);
        const auto &alph = alpha.const_array(mfi);
        const auto &gam  = gamma.const_array(mfi);
        const auto &src  = source.const_array(mfi);
        AMREX_D_TERM(const auto &bx = beta[0].const_array(mfi);
                     , const auto &by = beta[1].const_array(mfi);
                     , const auto &bz = beta[2].const_array(mfi);)
        For(sbx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                EXPECT_NEAR(residual(i, j, k, a, b, 0, alph,
                                     AMREX_D_DECL(bx, by, bz), gam, src, out,
                                     dx),
                            0, rhs_norm * 1e-12);
            });
    }
}

TEST_F(MLABecLapGradTest, UnsplitGridNoGrad)
{
    setup(false, 1, 2, 0);

    MLMG solver(lapgrad);
    solver.setMaxIter(100);
    solver.setMaxFmgIter(0);
    solver.setVerbose(1);
    solver.setBottomVerbose(1);

    const Real TOL_RES = 1e-12;
    const Real TOL_ABS = 0;

    solver.solve({ &stateout }, { &source }, TOL_RES, TOL_ABS);

    for (MFIter mfi(stateout); mfi.isValid(); ++mfi)
    {
        // check values without boundary conditions
        const auto &sbx  = amrex::grow(mfi.validbox(), -1);
        const auto &out  = stateout.const_array(mfi);
        const auto &alph = alpha.const_array(mfi);
        const auto &gam  = gamma.const_array(mfi);
        const auto &src  = source.const_array(mfi);
        AMREX_D_TERM(const auto &bx = beta[0].const_array(mfi);
                     , const auto &by = beta[1].const_array(mfi);
                     , const auto &bz = beta[2].const_array(mfi);)
        For(sbx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                EXPECT_NEAR(residual(i, j, k, a, b, c, alph,
                                     AMREX_D_DECL(bx, by, bz), gam, src, out,
                                     dx),
                            0, rhs_norm * 1e-12);
            });
    }
}

TEST_F(MLABecLapGradTest, UnsplitGridAllTerms)
{
    setup(false, 1, 2, -4);

    MLMG solver(lapgrad);
    solver.setMaxIter(100);
    solver.setMaxFmgIter(0);
    solver.setVerbose(1);
    solver.setBottomVerbose(1);

    const Real TOL_RES = 1e-12;
    const Real TOL_ABS = 0;

    solver.solve({ &stateout }, { &source }, TOL_RES, TOL_ABS);

    for (MFIter mfi(stateout); mfi.isValid(); ++mfi)
    {
        // check values without boundary conditions
        const auto &sbx  = amrex::grow(mfi.validbox(), -1);
        const auto &out  = stateout.const_array(mfi);
        const auto &alph = alpha.const_array(mfi);
        const auto &gam  = gamma.const_array(mfi);
        const auto &src  = source.const_array(mfi);
        AMREX_D_TERM(const auto &bx = beta[0].const_array(mfi);
                     , const auto &by = beta[1].const_array(mfi);
                     , const auto &bz = beta[2].const_array(mfi);)
        For(sbx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                EXPECT_NEAR(residual(i, j, k, a, b, c, alph,
                                     AMREX_D_DECL(bx, by, bz), gam, src, out,
                                     dx),
                            0, rhs_norm * 1e-12);
            });
    }
}