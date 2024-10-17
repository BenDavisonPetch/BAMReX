#ifndef MultiAdvAmrLevel_H_
#define MultiAdvAmrLevel_H_

#include "MultiAdvAmr/MultiAdvAmr.H"
#include <AMReX_AmrLevel.H>

namespace bamrex
{

using namespace amrex;

/**
 * \brief Virtual base class for managing individual levels with flexible
 * advance call structure.
 *
 * Used in conjunction with MultiAdvAmr.
 * TODO
 *
 * \sa AmrLevel
 */
class MultiAdvAmrLevel : public AmrLevel
{
  public:
    //! How many times to call advance on course level for a single time step
    virtual int NumAdvances() = 0;

    Real advance(Real, Real, int, int) override final
    {
        Abort("Calling the wrong advance routine! Use "
              "MultiAdvAmrLevel::advance(Real, Real, int, int, int) instead.");
        return 0.;
    };

    /**
     * \brief Do an integration step on this level. Returns maximum safe
     * timestep.
     *
     * \param time      initial time
     * \param dt        level time step
     * \param crse_iter ranges from 1 to NumAdvances(). Indicates how many
     *                  times advance has been called on the coarsest level
     * \param iteration ranges from 1 to ncycle. Indicates how many times
     *                  advance has been called since the last advance on
     *                  the level below
     * \param ncycle    number of time subcycles per timestep on the level
     *                  below
     */
    virtual Real advance(Real time, Real dt, int crse_iter, int iteration,
                         int ncycle)
        = 0;

  protected:
    MultiAdvAmrLevel() noexcept {} // NOLINT

    MultiAdvAmrLevel(Amr &papa, int lev, const Geometry &level_geom,
                     const BoxArray &ba, const DistributionMapping &dm,
                     Real time);
};

} // namespace bamrex

#endif