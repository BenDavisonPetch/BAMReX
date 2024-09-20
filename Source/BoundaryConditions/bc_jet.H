/**
 * Boundary conditions for a co-axial jet
 */

#ifndef bc_jet_H_
#define bc_jet_H_

#include "MultiBCFunct.H"

#include <AMReX_BCRec.H>
#include <AMReX_PhysBCFunct.H>
#include <AMReX_StateDescriptor.H>

//! For the conservative variables
amrex::BndryFuncFabDefault
make_coax_jetfill_bcfunc(const amrex::RealArray &center, amrex::Real radius_p,
                         amrex::Real radius_s, int wall, amrex::Real v_p,
                         amrex::Real v_s, amrex::Real T_p, amrex::Real T_s,
                         amrex::Real R_spec, amrex::Real p_amb,
                         amrex::Real adia, amrex::Real eps);

//! For the pressure
bamrex::MultiCCBndryFuncFabDefault make_coax_jetfill_bcfunc_p(
    const amrex::RealArray &center, amrex::Real radius_p, amrex::Real radius_s,
    int wall, amrex::Real T_p, amrex::Real T_s, amrex::Real R_spec,
    amrex::Real p_amb, amrex::Real adia, amrex::Real eps);

#endif