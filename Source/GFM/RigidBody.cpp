#include "RigidBody.H"
#include "BoundaryConditions/BCs.H"
#include "Euler/Euler.H"
#include "Euler/NComp.H"
#include "Euler/RiemannSolver.H"
#include "GFM/Extrapolate.H"
#include "GFM/GFMFlag.H"
#include "GFM/Interp_K.H"
#include "Index.H"
#include <AMReX_RealVect.H>

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

    const Real adia = AmrLevelAdv::h_parm->adiabatic;
    const Real eps  = AmrLevelAdv::h_parm->epsilon;

    const auto &dx  = geom.CellSizeArray();
    const auto &rdx = geom.InvCellSizeArray();

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
                    if (at_interface(flags(i, j, k))
                        && is_ghost(flags(i, j, k)))
                    {
                        // don't need prob_lo here because of how
                        // bilinear_interp works
                        const RealVect x_gh(AMREX_D_DECL((i + 0.5) * dx[0],
                                                         (j + 0.5) * dx[1],
                                                         (k + 0.5) * dx[2]));
                        const RealVect norm(AMREX_D_DECL(
                            ls(i, j, k, 1), ls(i, j, k, 2), ls(i, j, k, 3)));

                        if (rb_bc == RigidBodyBCType::reflective)
                        {
                            constexpr Real delta
                                = 1.5; // we always look delta*dx into the
                                       // fluid from the boundary to
                                       // interpolate
                            // normals point into fluid and ls >0 in body
                            const RealVect x_refl(
                                x_gh + (ls(i, j, k) + delta * dx[0]) * norm);
                            // const RealVect x_refl(
                            //     x_gh + 2*ls(i, j, k) * norm);
                            bilinear_interp(x_refl, arr, arr, i, j, k, dx,
                                            EULER_NCOMP);

                            const auto primv_L_Nd
                                = primv_from_consv(arr, adia, eps, i, j, k);
                            const Real u_n
                                = (AMREX_D_TERM(primv_L_Nd[1] * norm[0],
                                                +primv_L_Nd[2] * norm[1],
                                                +primv_L_Nd[3] * norm[2]));
                            const RealVect u_t(
                                AMREX_D_DECL(primv_L_Nd[1] - u_n * norm[0],
                                             primv_L_Nd[2] - u_n * norm[1],
                                             primv_L_Nd[3] - u_n * norm[2]));

                            // TODO: precompute this
                            const Real curvature = AMREX_D_TERM(
                                -(0.5 * rdx[0]
                                  * (ls(i + 1, j, k, 1) - ls(i - 1, j, k, 1))),
                                -(0.5 * rdx[1]
                                  * (ls(i, j + 1, k, 2) - ls(i, j - 1, k, 2))),
                                -(0.5 * rdx[2]
                                  * (ls(i, j, k + 1, 3)
                                     - ls(i, j, k - 1, 3))));

                            const GpuArray<Real, EULER_NCOMP> primv_L{
                                primv_L_Nd[0],
                                AMREX_D_DECL(-u_n, u_t.vectorLength(), 0),
                                primv_L_Nd[1 + AMREX_SPACEDIM]
                            };
                            // Account for pressure gradient

                            // dp/dn = density * tangential_vel^2 * curvature -
                            //     density * rigidbody_acceleration
                            const Real p_R = max(
                                primv_L_Nd[1 + AMREX_SPACEDIM]
                                    - primv_L_Nd[0] * u_t.radSquared()
                                          * curvature * 2 * delta * dx[0],
                                1e-14);
                            // Print() << "p_R = " << p_R << std::endl;
                            // Print() << "u_t.radSquared() = " <<
                            // u_t.radSquared() << std::endl; Print() <<
                            // "curvature = " << curvature << std::endl;

                            const GpuArray<Real, EULER_NCOMP> primv_R{
                                primv_L_Nd[0],
                                AMREX_D_DECL(u_n, u_t.vectorLength(), 0), p_R
                            };

                            EulerExactRiemannSolver<0> riem(primv_L, primv_R,
                                                            adia, adia, eps);
                            const auto consv_int_L = riem.get_interm_consv_L();
                            RealVect   q_int_L     = -consv_int_L[1] * norm
                                               + u_t * consv_int_L[0];

                            arr(i, j, k, 0) = consv_int_L[0];
                            for (int d = 0; d < AMREX_SPACEDIM; ++d)
                                arr(i, j, k, d + 1) = q_int_L[d];
                            arr(i, j, k, 1 + AMREX_SPACEDIM)
                                = consv_int_L[1 + AMREX_SPACEDIM];
                        }
                        else if (rb_bc == RigidBodyBCType::no_slip)
                        {
                            constexpr Real delta
                                = 1.5; // we always look delta*dx into the
                                       // fluid from the boundary to
                                       // interpolate
                            // normals point into fluid and ls >0 in body
                            const RealVect x_refl(
                                x_gh + (ls(i, j, k) + delta * dx[0]) * norm);
                            // const RealVect x_refl(
                            //     x_gh + 2*ls(i, j, k) * norm);
                            bilinear_interp(x_refl, arr, arr, i, j, k, dx,
                                            EULER_NCOMP);

                            auto primv_L_Nd
                                = primv_from_consv(arr, adia, eps, i, j, k);
                            AMREX_D_TERM(primv_L_Nd[QInd::U] *= -1;
                                         , primv_L_Nd[QInd::V] *= -1;
                                         , primv_L_Nd[QInd::W] *= -1;)
                            const auto consv_gh
                                = consv_from_primv(primv_L_Nd, adia, eps);
                            for (int n = 0; n < EULER_NCOMP; ++n)
                            {
                                arr(i, j, k, n) = consv_gh[n];
                            }
                        }
                        else if (rb_bc == RigidBodyBCType::foextrap)
                        {
                            constexpr Real delta
                                = 1.5; // we always look delta*dx into the
                                       // fluid from the boundary to
                                       // interpolate
                            // normals point into fluid and ls >0 in body
                            const RealVect x_refl(
                                x_gh + (ls(i, j, k) + delta * dx[0]) * norm);
                            // const RealVect x_refl(
                            //     x_gh + 2*ls(i, j, k) * norm);
                            bilinear_interp(x_refl, arr, arr, i, j, k, dx,
                                            EULER_NCOMP);
                        }
                    }
                });
        }
    }

    U.FillBoundary(geom.periodicity());
    extrapolate_from_boundary(geom, U, LS, gfm_flags, Vector<Real>(), 0,
                              U.nComp());
}

/**
 * \brief Fills pressure ghost cells for a rigid body defined by level set \c
 * LS
 *
 * \param geom
 * \param p
 * \param U Used to determine tangential velocity for slip condition
 * \param LS >0 -> body, =0 -> interface, <0 -> fluid. Components
 * 1:AMREX_SPACEDIM are the normal vectors (point into fluid, i.e. grad(LS))
 * \param gfm_flags constructed using build_gfm_flags
 * \param rb_bc
 */
AMREX_GPU_HOST void fill_ghost_p_rb(const amrex::Geometry           &geom,
                                    amrex::MultiFab                 &p,
                                    const amrex::MultiFab           &U,
                                    const amrex::MultiFab           &LS,
                                    const amrex::iMultiFab          &gfm_flags,
                                    RigidBodyBCType::rigidbodybctype rb_bc)
{
    return;
    AMREX_ASSERT_WITH_MESSAGE(
        LS.nComp() == AMREX_SPACEDIM + 1,
        "LS should store the level set and the normal vectors");
    AMREX_ASSERT(rb_bc == RigidBodyBCType::reflective
                 || rb_bc == RigidBodyBCType::no_slip);

    AMREX_ASSERT(p.nComp() == 1);

    const auto &dx  = geom.CellSizeArray();
    const auto &rdx = geom.InvCellSizeArray();

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    {
        for (MFIter mfi(p, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            // Fill on real box only (and assume that U has enough ghost
            // cells that the real cells needed to determine the ghost state
            // are always contained within the FAB)

            // TODO: skip FABs with no ghost cells
            const auto &bx    = mfi.tilebox();
            const auto &parr  = p.array(mfi);
            const auto &Uarr  = U.const_array(mfi);
            const auto &ls    = LS.const_array(mfi);
            const auto &flags = gfm_flags.const_array(mfi);
            ParallelFor(
                bx,
                [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    if (at_interface(flags(i, j, k))
                        && is_ghost(flags(i, j, k)))
                    {
                        // don't need prob_lo here because of how
                        // bilinear_interp works
                        const RealVect x_gh(AMREX_D_DECL((i + 0.5) * dx[0],
                                                         (j + 0.5) * dx[1],
                                                         (k + 0.5) * dx[2]));
                        const RealVect norm(AMREX_D_DECL(
                            ls(i, j, k, 1), ls(i, j, k, 2), ls(i, j, k, 3)));

                        if (rb_bc == RigidBodyBCType::reflective)
                        {
                            constexpr Real delta
                                = 1.5; // we always look delta*dx into the
                                       // fluid from the boundary to
                                       // interpolate
                            // normals point into fluid and ls >0 in body
                            const RealVect x_refl(
                                x_gh + (ls(i, j, k) + delta * dx[0]) * norm);

                            bilinear_interp(x_refl, parr, parr, i, j, k, dx,
                                            1);
                            const auto consv = bilinear_interp<EULER_NCOMP>(
                                x_refl, Uarr, dx);
                            const Real     q_n = (AMREX_D_TERM(
                                    consv[1] * norm[0], +consv[2] * norm[1],
                                    +consv[3] * norm[2]));
                            const RealVect u_t(AMREX_D_DECL(
                                (consv[1] - q_n * norm[0]) / consv[0],
                                (consv[2] - q_n * norm[1]) / consv[0],
                                (consv[3] - q_n * norm[2]) / consv[0]));

                            // TODO: precompute this
                            const Real curvature = AMREX_D_TERM(
                                -(0.5 * rdx[0]
                                  * (ls(i + 1, j, k, 1) - ls(i - 1, j, k, 1))),
                                -(0.5 * rdx[1]
                                  * (ls(i, j + 1, k, 2) - ls(i, j - 1, k, 2))),
                                -(0.5 * rdx[2]
                                  * (ls(i, j, k + 1, 3)
                                     - ls(i, j, k - 1, 3))));
                            // Account for pressure gradient

                            // dp/dn = density * tangential_vel^2 * curvature -
                            //     density * rigidbody_acceleration
                            parr(i, j, k)
                                = parr(i, j, k)
                                  - consv[0] * u_t.radSquared() * curvature
                                        * (ls(i, j, k) + delta * dx[0]);
                        }
                    }
                });
        }
    }

    p.FillBoundary(geom.periodicity());
    extrapolate_from_boundary(geom, p, LS, gfm_flags, Vector<Real>(), 0, 1);
}