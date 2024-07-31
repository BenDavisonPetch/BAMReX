#include "RigidBody.H"
#include "BCs.H"
#include "Euler/NComp.H"
#include "GFM/GFMFlag.H"
#include "GFM/Interp_K.H"

using namespace amrex;

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
                                  RigidBodyBCType::rigidbodybctype rb_bc)
{
    AMREX_ASSERT_WITH_MESSAGE(
        LS.nComp() == AMREX_SPACEDIM + 1,
        "LS should store the level set and the normal vectors");

    AMREX_ASSERT(U.nComp() == EULER_NCOMP);

    constexpr Real delta = 1.5; // we always look delta*dx into the fluid from
                                // the boundary to interpolate

    const auto &dx = geom.CellSizeArray();
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    {
        for (MFIter mfi(U, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            // Fill on real box only (and assume that U has enough ghost
            // cells that the real cells needed to determine the ghost state
            // are always contained within the FAB)

            // TODO: skip FABs with no ghost cells
            const auto &bx    = mfi.tilebox();
            const auto &arr   = U.array(mfi);
            const auto &ls    = LS.const_array(mfi);
            const auto &flags = gfm_flags.const_array(mfi);
            ParallelFor(
                bx,
                [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    if (needs_fill(flags(i, j, k)))
                    {
                        // don't need prob_lo here because of how
                        // bilinear_interp works
                        const RealVect x_gh(AMREX_D_DECL((i + 0.5) * dx[0],
                                                         (j + 0.5) * dx[1],
                                                         (k + 0.5) * dx[2]));
                        const RealVect norm(AMREX_D_DECL(
                            ls(i, j, k, 1), ls(i, j, k, 2), ls(i, j, k, 3)));
                        // normals point into fluid and ls >0 in body
                        const RealVect x_refl(
                            x_gh + (ls(i, j, k) + delta * dx[0]) * norm);
                        // const RealVect x_refl(
                        //     x_gh + 2*ls(i, j, k) * norm);
                        bilinear_interp(x_refl, arr, arr, i, j, k, dx,
                                        EULER_NCOMP);

                        if (rb_bc == RigidBodyBCType::reflective)
                        {
                            // Now we reflect the momentum in the normal
                            // direction q -> q - 2 (q.n)n
                            const Real tdot
                                = 2
                                  * (AMREX_D_TERM(arr(i, j, k, 1) * norm[0],
                                                  +arr(i, j, k, 2) * norm[1],
                                                  +arr(i, j, k, 3) * norm[2]));
                            for (int n = 0; n < AMREX_SPACEDIM; ++n)
                            {
                                arr(i, j, k, n + 1) -= tdot * norm[n];
                            }
                        }
                        else if (rb_bc == RigidBodyBCType::no_slip)
                        {
                            for (int n = 0; n < AMREX_SPACEDIM; ++n)
                            {
                                arr(i, j, k, n + 1) *= -1;
                            }
                        }
                    }
                });
        }
    }
}