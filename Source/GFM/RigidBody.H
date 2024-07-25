#ifndef RigidBody_H_
#define RigidBody_H_

#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>

/**
 * \brief Fills ghost cells for a rigid body defined by level set \c LS
 *
 * \param geom
 * \param U
 * \param LS >0 -> body, =0 -> interface, <0 -> fluid
 * \param gfm_flags constructed using build_gfm_flags
 */
AMREX_GPU_HOST void
fill_ghost_rb(const amrex::Geometry &geom, amrex::MultiFab &U,
              const amrex::MultiFab &LS, const amrex::iMultiFab &gfm_flags);

#endif