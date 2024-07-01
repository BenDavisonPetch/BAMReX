#include <AMReX_Box.H>
#include <gtest/gtest.h>

#include "AmrLevelAdv.H"
#include "Fluxes/Update.H"
#include "IMEX/IMEX_RK.H"

#include <AMReX_PROB_AMR_F.H>
#include <AMReX_Random.H>

using namespace amrex;

void randomise_state(MultiFab                        &statein,
                     Array<MultiFab, AMREX_SPACEDIM> &fluxes)
{
    for (MFIter mfi(statein, false); mfi.isValid(); ++mfi)
    {
        const auto &bx   = mfi.growntilebox();
        const auto &nbx0 = mfi.nodaltilebox(0);
        const auto &nbx1 = mfi.nodaltilebox(1);
        const auto &nbx2 = mfi.nodaltilebox(2);

        const auto &fx = fluxes[0].array(mfi);
        const auto &fy = fluxes[1].array(mfi);
        const auto &fz = fluxes[2].array(mfi);
        const auto &in = statein.array(mfi);

        For(
            nbx0, fx.nComp(),
            [=](int i, int j, int k, int n)
            { fx(i, j, k, n) = (Random() - 0.5) * 10; },
            nbx1, fy.nComp(),
            [=](int i, int j, int k, int n)
            { fy(i, j, k, n) = (Random() - 0.5) * 10; },
            nbx2, fz.nComp(),
            [=](int i, int j, int k, int n)
            { fz(i, j, k, n) = (Random() - 0.5) * 10; });
        For(bx, in.nComp(),
            [=](int i, int j, int k, int n)
            { in(i, j, k, n) = (Random() - 0.5) * 10; });
    }
}

class UpdateTest : public testing::Test
{
  protected:
    amrex::IntVect                               dom_lo, dom_hi;
    amrex::Box                                   domain;
    amrex::BoxArray                              ba;
    amrex::DistributionMapping                   dm;
    amrex::MultiFab                              mf;
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo, prob_hi;
    amrex::RealBox                               real_box;
    amrex::Geometry                              geom;
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx;
    Vector<Array<MultiFab, AMREX_SPACEDIM> >     fluxes;
    MultiFab                                     statein;

    UpdateTest() {}

    void setup(int nx, int ny, int nz, int n_fluxes, int max_grid_size = -1)
    {
        InitRandom(31521);
        dom_lo = IntVect(AMREX_D_DECL(0, 0, 0));
        dom_hi = IntVect(AMREX_D_DECL(nx - 1, ny - 1, nz - 1));
        domain = Box(dom_lo, dom_hi);
        ba     = BoxArray(domain);
        if (max_grid_size > 0)
            ba.maxSize(max_grid_size);
        dm       = DistributionMapping(ba);
        prob_lo  = { AMREX_D_DECL(0.0, 0.0, 0.0) };
        prob_hi  = { AMREX_D_DECL(1.0, 1.0, 1.0) };
        real_box = RealBox(AMREX_D_DECL(0.0, 0.0, 0.0),
                           AMREX_D_DECL(1.0, 1.0, 1.0));
        geom     = Geometry(domain, &real_box);
        dx       = geom.CellSizeArray();

        AmrLevelAdv::h_prob_parm = new ProbParm{};
        int       dummy          = 0;
        Real      dummyreal      = 0;
        ParmParse pp("prob");
        pp.add("adiabatic", 1.4);
        pp.add("epsilon", 1);
        amrex_probinit(&dummy, &dummy, &dummy, &dummyreal, &dummyreal);

        statein.define(ba, dm, EULER_NCOMP, 2);
        fluxes.resize(n_fluxes);
        for (int d = 0; d < AMREX_SPACEDIM; ++d)
        {
            BoxArray ba_ = ba;
            ba_.surroundingNodes(d);
            for (int n = 0; n < n_fluxes; ++n)
                fluxes[n][d].define(ba_, dm, EULER_NCOMP, 0);
        }
        for (int n = 0; n < n_fluxes; ++n) randomise_state(statein, fluxes[n]);
    }
};
class ConservativeUpdate : public UpdateTest
{
};

#if AMREX_SPACEDIM != 3
#error "3D tests only!"
#endif

TEST_F(ConservativeUpdate, NormalSameAsWeightedVersion)
{
    setup(10, 5, 2, 3, 5); // dx = (0.1, 0.2, 0.5)
    const Vector<Real> dts = { 0.1, 0.2, 0.6 };
    MultiFab           tmp1(ba, dm, EULER_NCOMP, 0);
    MultiFab           tmp2(ba, dm, EULER_NCOMP, 0);
    tmp1.setVal(0.5);
    tmp2.setVal(0.25);

    MultiFab stateout1(ba, dm, EULER_NCOMP, 0);
    MultiFab stateout2(ba, dm, EULER_NCOMP, 0);
    stateout1.setVal(1);
    stateout2.setVal(-1);

    conservative_update(geom, statein, fluxes[0], tmp1, dts[0]);
    conservative_update(geom, tmp1, fluxes[1], tmp2, dts[1]);
    conservative_update(geom, tmp2, fluxes[2], stateout1, dts[2]);

    conservative_update(geom, statein, fluxes, stateout2, dts, 3);

    for (MFIter mfi(stateout1, false); mfi.isValid(); ++mfi)
    {
        const auto &bx   = mfi.validbox();
        const auto &out1 = stateout1.const_array(mfi);
        const auto &out2 = stateout2.const_array(mfi);
        For(bx, stateout1.nComp(),
            [=](int i, int j, int k, int n)
            {
                EXPECT_NEAR(out1(i, j, k, n), out2(i, j, k, n),
                            fabs(out1(i, j, k, n) * 1e-12));
            });
    }
}

TEST_F(ConservativeUpdate, StateCopying)
{
    // If N = 0 state should be copied
    setup(10, 5, 2, 2, 5); // dx = (0.1, 0.2, 0.5)

    MultiFab stateout(ba, dm, EULER_NCOMP, 0);
    stateout.setVal(2);
    conservative_update(geom, statein, fluxes, stateout, { 0.1, 0.5 }, 0);

    for (MFIter mfi(stateout, false); mfi.isValid(); ++mfi)
    {
        const auto &bx  = mfi.validbox();
        const auto &out = stateout.const_array(mfi);
        const auto &in  = statein.const_array(mfi);
        For(bx, stateout.nComp(),
            [=](int i, int j, int k, int n)
            {
                EXPECT_NEAR(out(i, j, k, n), in(i, j, k, n),
                            fabs(in(i, j, k, n) * 1e-12));
            });
    }
}

TEST_F(ConservativeUpdate, SumFluxes)
{
    setup(10, 5, 2, 3, 5);
    Array<MultiFab, AMREX_SPACEDIM> summed_fluxes;
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        auto nodeba = ba;
        nodeba.surroundingNodes(d);
        summed_fluxes[d].define(nodeba, dm, EULER_NCOMP, 0);
        summed_fluxes[d].setVal(d);
    }
    const Vector<Real> weighting = { 1, 2, -3 };
    const Vector<Real> dts       = { 0.1, 0.2, -0.3 };
    const Real         dt        = 0.1;

    MultiFab stateout1(ba, dm, EULER_NCOMP, 0);
    MultiFab stateout2(ba, dm, EULER_NCOMP, 0);
    stateout1.setVal(0.5);
    stateout2.setVal(5);

    Print() << "Doing first update" << std::endl;

    conservative_update(geom, statein, fluxes, stateout1, dts, 3);
    Print() << "Completed first update" << std::endl;
    sum_fluxes(fluxes, summed_fluxes, weighting);
    conservative_update(geom, statein, summed_fluxes, stateout2, dt);

    for (MFIter mfi(stateout1, false); mfi.isValid(); ++mfi)
    {
        const auto &bx   = mfi.validbox();
        const auto &out1 = stateout1.const_array(mfi);
        const auto &out2 = stateout2.const_array(mfi);
        For(bx, stateout1.nComp(),
            [=](int i, int j, int k, int n)
            {
                EXPECT_NEAR(out1(i, j, k, n), out2(i, j, k, n),
                            fabs(out1(i, j, k, n) * 1e-12));
            });
    }
}