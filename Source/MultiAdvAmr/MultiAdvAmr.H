#ifndef MultiAdvAmr_H_
#define MultiAdvAmr_H_

#include <AMReX_Amr.H>

namespace bamrex
{
using namespace amrex;

/**
 * \brief Manage hierarchy of levels for time-dependent AMR computations, with
 * a flexible time-step call structure for individual levels.
 *
 * Each recursive stack of advance() calls across the levels (as is done by
 * amrex::Amr()) can be done multiple times in a single time step, specified by
 * MultiAdvAmrLevel::NumAdvances(). For example, with NumAdvances=2, advance()
 * is called twice on the coarsest level, with the same start time.
 *
 * \sa amrex::Amr
 */
class MultiAdvAmr : public Amr
{
  public:
    //! The constructor
    MultiAdvAmr(LevelBld *a_levelbld);

    MultiAdvAmr(const RealBox *rb, int max_level_in,
                const Vector<int> &n_cell_in, int coord, LevelBld *a_levelbld);

    MultiAdvAmr(const MultiAdvAmr &rhs) = delete;
    MultiAdvAmr(MultiAdvAmr &&rhs)      = delete;

    void InitAmr();

    static void Initialize();
    static void Finalize();

  protected:
    virtual void timeStep(int level, Real time, int crse_iter, int iteration,
                          int niter, Real stop_time);
    // Called by Amr::coarseTimeStep on level 0
    virtual void timeStep(int level, Real time, int iteration, int niter,
                          Real stop_time) override;

    //
    // The data...
    //
    //! If true the grid is restored to its initial state before starting a new
    //! advance stack. Defaults to false
    bool restore_grid_on_crse_iter;
    //! If false regridding will only occur before a coarse time step. Defaults
    //! to true
    bool regrid_during_timestep;
};

} // namespace bamrex

#endif