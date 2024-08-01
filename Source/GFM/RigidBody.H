#ifndef RigidBody_H_
#define RigidBody_H_

#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>

#include "BCs.H"

/**
 * \brief Fills ghost cells for a rigid body defined by level set \c LS
 *
 * \param geom
 * \param U
 * \param LS >0 -> body, =0 -> interface, <0 -> fluid. Components
 * 1:AMREX_SPACEDIM are the normal vectors (point into fluid, i.e. grad(LS))
 * \param gfm_flags constructed using build_gfm_flags
 * \param rb_bc
 */
AMREX_GPU_HOST void fill_ghost_rb(const amrex::Geometry           &geom,
                                  amrex::MultiFab                 &U,
                                  const amrex::MultiFab           &LS,
                                  const amrex::iMultiFab          &gfm_flags,
                                  RigidBodyBCType::rigidbodybctype rb_bc);

/**
 * \brief Fills pressure ghost cells for a rigid body defined by level set \c
 * LS
 *
 * \param geom
 * \param p
 * \param LS >0 -> body, =0 -> interface, <0 -> fluid. Components
 * 1:AMREX_SPACEDIM are the normal vectors (point into fluid, i.e. grad(LS))
 * \param gfm_flags constructed using build_gfm_flags
 * \param rb_bc
 */
AMREX_GPU_HOST void fill_ghost_p_rb(const amrex::Geometry           &geom,
                                    amrex::MultiFab                 &p,
                                    const amrex::MultiFab           &LS,
                                    const amrex::iMultiFab          &gfm_flags,
                                    RigidBodyBCType::rigidbodybctype rb_bc);

#endif