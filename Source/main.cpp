
#include <iomanip>
#include <iostream>
#include <new>

#include <AMReX_Amr.H>
#include <AMReX_AmrLevel.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>

#include "AmrLevelAdv.H"
#include "Geometry/MakeSDF.H"

using namespace amrex;

amrex::LevelBld *getLevelBld();

int main(int argc, char *argv[])
{
    amrex::Initialize(argc, argv);

    auto dRunTime1 = amrex::second();

    int          max_step;
    Real         strt_time;
    Real         stop_time;
    Vector<Real> plot_times;

    {
        ParmParse pp;

        max_step  = -1;
        strt_time = 0.0;
        stop_time = -1.0;

        pp.query("max_step", max_step);
        pp.query("strt_time", strt_time);
        pp.query("stop_time", stop_time);

        ParmParse ppamr("amr");
        ppamr.queryarr("plot_at", plot_times);
        plot_times.push_back(stop_time);
        std::sort(plot_times.begin(), plot_times.end());
    }

    if (strt_time < 0.0)
    {
        amrex::Abort("MUST SPECIFY a non-negative strt_time");
    }

    if (max_step < 0 && stop_time < 0.0)
    {
        amrex::Abort(
            "Exiting because neither max_step nor stop_time is non-negative.");
    }

    {
        Amr amr(getLevelBld());

#ifdef AMREX_USE_EB
        AmrLevel::SetEBSupportLevel(EBSupport::full);
        // EB only used by linear solver
        AmrLevel::SetEBMaxGrowCells(AmrLevelAdv::NUM_GROW, 2, 2);
        // Initialize geometric database
        auto sdf   = SDF::make_sdf();
        auto gshop = EB2::makeShop(sdf);
        EB2::Build(gshop, amr.Geom(amr.maxLevel()), amr.maxLevel(),
                   amr.maxLevel());
#endif

        amr.init(strt_time, stop_time);

        int  iplot_times    = 0;
        Real next_stop_time = plot_times[iplot_times];

        while (amr.okToContinue()
               && (amr.levelSteps(0) < max_step || max_step < 0)
               && (amr.cumTime() < stop_time || stop_time < 0.0))

        {
            //
            // Do a coarse timestep.  Recursively calls timeStep()
            //
            amr.coarseTimeStep(next_stop_time);
            if (amr.cumTime() >= next_stop_time * (1 - 1e-12))
            {
                iplot_times++;
                if (iplot_times >= plot_times.size())
                    continue;
                next_stop_time = plot_times[iplot_times];
                if (amr.stepOfLastPlotFile() < amr.levelSteps(0))
                    amr.writePlotFile();
            }
        }

        // Write final checkpoint and plotfile
        if (amr.stepOfLastCheckPoint() < amr.levelSteps(0))
        {
            amr.checkPoint();
        }

        if (amr.stepOfLastPlotFile() < amr.levelSteps(0))
        {
            amr.writePlotFile();
        }
    }

    auto dRunTime2 = amrex::second() - dRunTime1;

    ParallelDescriptor::ReduceRealMax(dRunTime2,
                                      ParallelDescriptor::IOProcessorNumber());

    amrex::Print() << "Run time = " << dRunTime2 << std::endl;

    amrex::Finalize();

    return 0;
}
