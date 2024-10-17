#include "MultiAdvAmr.H"
#include "MultiAdvAmrLevel.H"

#include <AMReX_ParmParse.H>

namespace bamrex
{
using namespace amrex;

namespace
{
bool initialized = false;

//
// ParmParse variables from amrex::Amr that are needed for timeStep
//
bool regrid_on_restart;
bool force_regrid_level_zero;
bool plotfile_on_restart;
}

void MultiAdvAmr::Initialize()
{
    if (initialized)
    {
        return;
    }
    //
    // Set defaults
    //

    // Same defaults as for amrex::Amr
    regrid_on_restart       = false;
    force_regrid_level_zero = false;
    plotfile_on_restart     = false;

    amrex::ExecOnFinalize(MultiAdvAmr::Finalize);

    initialized = true;
}

void MultiAdvAmr::Finalize() { initialized = false; }

MultiAdvAmr::MultiAdvAmr(LevelBld *a_levelbld)
    : Amr(a_levelbld)
{
    Initialize();
    InitAmr();
}

MultiAdvAmr::MultiAdvAmr(const RealBox *rb, int max_level_in,
                         const Vector<int> &n_cell_in, int coord,
                         LevelBld *a_levelbld)
    : Amr(rb, max_level_in, n_cell_in, coord, a_levelbld)
{
    Initialize();
    InitAmr();
}

void MultiAdvAmr::InitAmr()
{
    BL_PROFILE("MultiAdvAmr::InitAmr()");

    //
    // Set defaults
    //
    restore_grid_on_crse_iter = false;
    regrid_during_timestep    = true;

    ParmParse pp("amr");
    pp.queryAdd("restore_grid_on_crse_iter", restore_grid_on_crse_iter);
    pp.queryAdd("regrid_during_timestep", regrid_during_timestep);
    if (max_level > 0 && restore_grid_on_crse_iter)
        amrex::Error("Grid restoration on coarse iteration has not been "
                     "implemented"); // sorry
}

void MultiAdvAmr::timeStep(int level, Real time, int crse_iter, int iteration,
                           int niter, Real stop_time)
{
    BL_PROFILE("MultiAdvAmr::timeStep()");
    BL_COMM_PROFILE_NAMETAG("MultiAdvAmr::timeStep TOP");

    if (!(niter == 1
          || static_cast<MultiAdvAmrLevel *>(amr_level[level].get())
                     ->NumAdvances()
                 == 1))
        amrex::Error("Regridding with more than one advance sweeps and time "
                     "subcycling is not implemented.");

    // This is used so that the AmrLevel functions can know which level is
    // being advanced
    //      when regridding is called with possible lbase > level.
    which_level_being_advanced = level;

    // Update so that by default, we don't force a post-step regrid.
    amr_level[level]->setPostStepRegrid(0);

    if (max_level == 0 && force_regrid_level_zero)
    {
        regrid_level_0_on_restart();
    }

    //
    // Allow regridding of level 0 calculation on restart.
    //
    if (max_level == 0 && regrid_on_restart)
    {
        regrid_level_0_on_restart();
    }
    else if (regrid_during_timestep || (level == 0 && crse_iter == 0))
    {
        int lev_top = std::min(finest_level, max_level - 1);
        for (int i(level); i <= lev_top; ++i)
        {
            const int old_finest = finest_level;

            // checks if regridding is enabled and regrid_int is satisified
            if (okToRegrid(i))
            {
                regrid(i, time);

                //
                // Compute new dt after regrid if at level 0 and
                // compute_new_dt_on_regrid.
                //
                if (compute_new_dt_on_regrid && (i == 0))
                {
                    int post_regrid_flag = 1;
                    amr_level[0]->computeNewDt(
                        finest_level, sub_cycle, n_cycle, ref_ratio, dt_min,
                        dt_level, stop_time, post_regrid_flag);
                }

                for (int k(i); k <= finest_level; ++k)
                {
                    level_count[k] = 0;
                }

                if (old_finest < finest_level)
                {
                    //
                    // The new levels will not have valid time steps
                    // and iteration counts.
                    //
                    for (int k(old_finest + 1); k <= finest_level; ++k)
                    {
                        dt_level[k]
                            = dt_level[k - 1] / static_cast<Real>(n_cycle[k]);
                    }
                }
            }
            if (old_finest > finest_level)
            {
                lev_top = std::min(finest_level, max_level - 1);
            }
        }

        if (max_level == 0 && loadbalance_level0_int > 0
            && loadbalance_with_workestimates)
        {
            if (level_steps[0] == 1
                || level_count[0] >= loadbalance_level0_int)
            {
                LoadBalanceLevel0(time);
                level_count[0] = 0;
            }
        }
    }
    //
    // Check to see if should write plotfile.
    // This routine is here so it is done after the restart regrid.
    //
    if (plotfile_on_restart && !(restart_chkfile.empty()))
    {
        plotfile_on_restart = false;
        writePlotFile();
    }
    //
    // Advance grids at this level.
    //
    if (verbose > 0)
    {
        amrex::Print() << "[Level " << level << " step "
                       << level_steps[level] + 1 << "] "
                       << "ADVANCE " << crse_iter << " at time " << time
                       << " with dt = " << dt_level[level] << "\n";
    }

    Real dt_new
        = static_cast<MultiAdvAmrLevel *>(amr_level[level].get())
              ->advance(time, dt_level[level], crse_iter, iteration, niter);
    BL_PROFILE_REGION_STOP("amr_level.advance");

    dt_min[level] = (iteration == 1 && crse_iter == 1)
                        ? dt_new
                        : std::min(dt_min[level], dt_new);

    if (crse_iter == 1)
    {
        level_steps[level]++;
        level_count[level]++;
    }

    if (verbose > 0)
    {
        amrex::Print() << "[Level " << level << " step " << level_steps[level]
                       << "] "
                       << "Advanced " << amr_level[level]->countCells()
                       << " cells\n";
    }

    // If the level signified that it wants a regrid after the advance has
    // occurred, do that now.
    if (amr_level[level]->postStepRegrid())
    {

        int old_finest = finest_level;

        regrid(level, time);

        if (old_finest < finest_level)
        {
            //
            // The new levels will not have valid time steps.
            //
            for (int k = old_finest + 1; k <= finest_level; ++k)
            {
                dt_level[k] = dt_level[k - 1] / static_cast<Real>(n_cycle[k]);
            }
        }
    }

    //
    // Advance grids at higher level.
    //
    if (level < finest_level)
    {
        const int lev_fine = level + 1;

        if (sub_cycle)
        {
            const int ncycle = n_cycle[lev_fine];

            BL_COMM_PROFILE_NAMETAG("MultiAdvAmr::timeStep timeStep subcycle");
            for (int i = 1; i <= ncycle; i++)
            {
                timeStep(lev_fine,
                         time + static_cast<Real>(i - 1) * dt_level[lev_fine],
                         crse_iter, i, ncycle, stop_time);
            }
        }
        else
        {
            BL_COMM_PROFILE_NAMETAG(
                "MultiAdvAmr::timeStep timeStep nosubcycle");
            timeStep(lev_fine, time, crse_iter, 1, 1, stop_time);
        }
    }

    amr_level[level]->post_timestep(iteration);

    // Set this back to negative so we know whether we are in fact in this
    // routine
    which_level_being_advanced = -1;
}

// This function is used to intercept amrex::Amr's initiation of the advance
// sweep, so that we can do more than one
void MultiAdvAmr::timeStep(int level, Real time, int iteration, int niter,
                           Real stop_time)
{
    BL_PROFILE("MultiAdvAmr::timeStep()");
    BL_COMM_PROFILE_NAMETAG("MultiAdvAmr::timeStep TOP");

    AMREX_ASSERT(level == 0);
    const int nadv
        = static_cast<MultiAdvAmrLevel *>(amr_level[0].get())->NumAdvances();
    for (int crse_iter = 1; crse_iter <= nadv; ++crse_iter)
    {
        timeStep(level, time, crse_iter, iteration, niter, stop_time);
    }
}

} // namespace bamrex