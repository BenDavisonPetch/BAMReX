/**
 * Generate the SDF from the input file
 */
#ifndef MakeSDF_H_
#define MakeSDF_H_

#include <functional>

#include <AMReX_Array.H>
#include <AMReX_REAL.H>
#include <AMReX_SPACE.H>

namespace SDF
{

using SDF = std::function<amrex::Real(amrex::RealArray)>;

/**
 * Construct an SDF from ParmParse inputs
 */
SDF make_sdf();

} // namespace SDF

#endif