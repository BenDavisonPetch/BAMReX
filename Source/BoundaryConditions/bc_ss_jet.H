/**
 * Boundary conditions for a supersonic jet
 */

#ifndef bc_ss_jet_H_
#define bc_ss_jet_H_

#include "MultiBCFunct.H"

#include <AMReX_BCRec.H>
#include <AMReX_PhysBCFunct.H>
#include <AMReX_StateDescriptor.H>

//! For the conservative variables
amrex::BndryFuncFabDefault
make_ss_jetfill_bcfunc(const amrex::RealArray &center, amrex::Real radius,
                       int wall, amrex::Real M, amrex::Real rho_amb,
                       amrex::Real p_amb);

#endif