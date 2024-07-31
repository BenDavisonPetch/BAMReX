/**
 * Boundary conditions for a co-axial jet
 */

#ifndef bc_jet_H_
#define bc_jet_H_

#include <AMReX_PhysBCFunct.H>
#include <AMReX_StateDescriptor.H>

amrex::BndryFuncFabDefault
make_coax_jetfill_bcfunc(const amrex::RealArray &center, amrex::Real radius_p,
                         amrex::Real radius_s, int wall, amrex::Real v_p,
                         amrex::Real v_s, amrex::Real T_p, amrex::Real T_s,
                         amrex::Real R_spec, amrex::Real p_amb,
                         amrex::Real adia, amrex::Real eps);

#endif