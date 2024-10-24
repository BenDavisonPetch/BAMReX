#include "Prob.H"

#include <AMReX_MultiFab.H>

/**
 * Initialize Data on Multifab
 *
 * \param S_tmp Pointer to Multifab where data is to be initialized.
 * \param geom Pointer to Multifab's geometry data.
 *
 */
using namespace amrex;

void initdata(MultiFab &S_tmp, const Geometry &geom)
{
    S_tmp.setVal(1);
    amrex::ignore_unused(geom);
}