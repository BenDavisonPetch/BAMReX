#include "MultiAdvAmrLevel.H"

using namespace amrex;
using namespace bamrex;

MultiAdvAmrLevel::MultiAdvAmrLevel(Amr &papa, int lev,
                                   const Geometry            &level_geom,
                                   const BoxArray            &ba,
                                   const DistributionMapping &dm, Real time)
    : AmrLevel(papa, lev, level_geom, ba, dm, time)
{
}