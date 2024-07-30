/**
 * Boundary conditions for a co-axial jet
 */

#ifndef bc_jet_H_
#define bc_jet_H_

#include <AMReX_PhysBCFunct.H>
#include <AMReX_StateDescriptor.H>

amrex::BndryFuncFabDefault
make_coax_jetfill_bcfunc(const amrex::RealArray &center, amrex::Real radius_p,
                         amrex::Real radius_s, int wall, amrex::Real dens,
                         amrex::Real vel_p, amrex::Real vel_s, amrex::Real p,
                         amrex::Real adia, amrex::Real eps);

#endif