#ifndef AMREX_MLABECLAP_1D_K_H_
#define AMREX_MLABECLAP_1D_K_H_
#include <AMReX_Config.H>
#include <AMReX_FArrayBox.H>

namespace amrex
{

template <typename T>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
mlalapgrad_adotx(int i, int, int, int n, Array4<T> const &y,
                 Array4<T const> const &x, Array4<T const> const &a,
                 Array4<T const> const &c, Array4<T const> const &bX,
                 GpuArray<T, AMREX_SPACEDIM> const &dxinv, T alpha, T beta,
                 T gamma) noexcept
{
    const T dhx  = beta * dxinv[0] * dxinv[0];
    const T chdx = gamma * c(i, 0, 0, 0) * 0.5 * dxinv[0];
    y(i, 0, 0, n)
        = alpha * a(i, 0, 0) * x(i, 0, 0, n)
          - dhx
                * (bX(i + 1, 0, 0, n) * (x(i + 1, 0, 0, n) - x(i, 0, 0, n))
                   - bX(i, 0, 0, n) * (x(i, 0, 0, n) - x(i - 1, 0, 0, n)))
          + chdx * (x(i + 1, 0, 0, n) - x(i - 1, 0, 0, n));
}

// template <typename T>
// AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
// void mlabeclap_adotx_os (int i, int, int, int n, Array4<T> const& y,
//                          Array4<T const> const& x,
//                          Array4<T const> const& a,
//                          Array4<T const> const& bX,
//                          Array4<int const> const& osm,
//                          GpuArray<T,AMREX_SPACEDIM> const& dxinv,
//                          T alpha, T beta) noexcept
// {
//     const T dhx = beta*dxinv[0]*dxinv[0];
//     if (osm(i,0,0) == 0) {
//         y(i,0,0,n) = T(0.0);
//     } else {
//         y(i,0,0,n) = alpha*a(i,0,0)*x(i,0,0,n)
//             - dhx * (bX(i+1,0,0,n)*(x(i+1,0,0,n) - x(i  ,0,0,n))
//                    - bX(i  ,0,0,n)*(x(i  ,0,0,n) - x(i-1,0,0,n)));
//     }
// }

template <typename T>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
mlalapgrad_normalize(int i, int, int, int n, Array4<T> const &x,
                     Array4<T const> const &a, Array4<T const> const &bX,
                     GpuArray<T, AMREX_SPACEDIM> const &dxinv, T alpha,
                     T beta) noexcept
{
    const T dhx = beta * dxinv[0] * dxinv[0];
    x(i, 0, 0, n)
        /= alpha * a(i, 0, 0) + dhx * (bX(i, 0, 0, n) + bX(i + 1, 0, 0, n));
}

// template <typename T>
// AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
// void mlabeclap_flux_x (Box const& box, Array4<T> const& fx, Array4<T const>
// const& sol,
//                        Array4<T const> const& bx, T fac, int ncomp) noexcept
// {
//     const auto lo = amrex::lbound(box);
//     const auto hi = amrex::ubound(box);

//     for (int n = 0; n < ncomp; ++n) {
//     AMREX_PRAGMA_SIMD
//     for (int i = lo.x; i <= hi.x; ++i) {
//         fx(i,0,0,n) = -fac*bx(i,0,0,n)*(sol(i,0,0,n)-sol(i-1,0,0,n));
//     }
//     }
// }

// // Only compute fluxes at lo and hi domain x faces
// template <typename T>
// AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
// void mlabeclap_flux_xface (Box const& box, Array4<T> const& fx, Array4<T
// const> const& sol,
//                            Array4<T const> const& bx, T fac, int xlen, int
//                            ncomp) noexcept
// {
//     const auto lo = amrex::lbound(box);

//     for (int n = 0; n < ncomp; ++n) {
//     int i = lo.x;
//     fx(i,0,0,n) = -fac*bx(i,0,0,n)*(sol(i,0,0,n)-sol(i-1,0,0,n));
//     i += xlen;
//     fx(i,0,0,n) = -fac*bx(i,0,0,n)*(sol(i,0,0,n)-sol(i-1,0,0,n));
//     }
// }

/**
 * Red-black Gauss Seidel sweep for
 * (A * a - B div( b grad ) + C c . grad) phi = rhs
 * c is a d-dim vector
 *
 * \param i
 * \param n component of linear system being solved
 * \param phi solution to be updated with GS sweep
 * \param rhs RHS of equation
 * \param alpha A from eqn above
 * \param a a from eqn above
 * \param c c from eqn above
 * \param dhx B/dx^2
 * \param cdhx C / (2dx)
 * \param bX x aligned face centred values of b from eqn above
 * \param m0 a mask of whether cells are covered or outside the domain - I
 * don't really get it
 * \param m1 another mask, again not sure
 * \param f0 under relaxation values - something to do with interpolation
 * \param f1 same again
 * \param vbox not sure
 * \param redblack gives whether iteration is even or odd
 */
template <typename T>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void alapgrad_gsrb(
    int i, int, int, int n, Array4<T> const &phi, Array4<T const> const &rhs,
    T alpha, Array4<T const> const &a, Array4<T const> const &c, T dhx, T cdhx,
    Array4<T const> const &bX, Array4<int const> const &m0,
    Array4<int const> const &m1, Array4<T const> const &f0,
    Array4<T const> const &f1, Box const &vbox, int redblack) noexcept
{
    // redblack is 0 if it's an even iteration and 1 if it's an odd iteration
    // on even iterations we only update even indices and on odd
    // iteration we only update odd indices
    if ((i + redblack) % 2 == 0)
    {
        const auto vlo = amrex::lbound(vbox);
        const auto vhi = amrex::ubound(vbox);

        T cf0 = (i == vlo.x && m0(vlo.x - 1, 0, 0) > 0) ? f0(vlo.x, 0, 0, n)
                                                        : T(0.0);
        T cf1 = (i == vhi.x && m1(vhi.x + 1, 0, 0) > 0) ? f1(vhi.x, 0, 0, n)
                                                        : T(0.0);

        // cf1 is hi and cf0 is lo

        T delta = dhx * (bX(i, 0, 0, n) * cf0 + bX(i + 1, 0, 0, n) * cf1)
                  + cdhx * c(i, 0, 0, 0) * (cf0 - cf1);

        T gamma
            = alpha * a(i, 0, 0) + dhx * (bX(i, 0, 0, n) + bX(i + 1, 0, 0, n));

        T rho = dhx
                    * (bX(i, 0, 0, n) * phi(i - 1, 0, 0, n)
                       + bX(i + 1, 0, 0, n) * phi(i + 1, 0, 0, n))
                - cdhx * c(i, 0, 0, 0)
                      * (phi(i + 1, 0, 0, n) - phi(i - 1, 0, 0, n));

        phi(i, 0, 0, n) = (rhs(i, 0, 0, n) + rho - phi(i, 0, 0, n) * delta)
                          / (gamma - delta);
    }
}

// template <typename T>
// AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
// void abec_gsrb_os (int i, int, int, int n, Array4<T> const& phi, Array4<T
// const> const& rhs,
//                    T alpha, Array4<T const> const& a,
//                    T dhx,
//                    Array4<T const> const& bX,
//                    Array4<int const> const& m0,
//                    Array4<int const> const& m1,
//                    Array4<T const> const& f0,
//                    Array4<T const> const& f1,
//                    Array4<int const> const& osm,
//                    Box const& vbox, int redblack) noexcept
// {
//     if ((i+redblack)%2 == 0) {
//         if (osm(i,0,0) == 0) {
//             phi(i,0,0) = T(0.0);
//         } else {
//             const auto vlo = amrex::lbound(vbox);
//             const auto vhi = amrex::ubound(vbox);

//             T cf0 = (i == vlo.x && m0(vlo.x-1,0,0) > 0)
//                 ? f0(vlo.x,0,0,n) : T(0.0);
//             T cf1 = (i == vhi.x && m1(vhi.x+1,0,0) > 0)
//                 ? f1(vhi.x,0,0,n) : T(0.0);

//             T delta = dhx*(bX(i,0,0,n)*cf0 + bX(i+1,0,0,n)*cf1);

//             T gamma = alpha*a(i,0,0)
//                 +   dhx*( bX(i,0,0,n) + bX(i+1,0,0,n) );

//             T rho = dhx*(bX(i  ,0  ,0,n)*phi(i-1,0  ,0,n)
//                             + bX(i+1,0  ,0,n)*phi(i+1,0  ,0,n));

//             phi(i,0,0,n) = (rhs(i,0,0,n) + rho - phi(i,0,0,n)*delta)
//                 / (gamma - delta);
//         }
//     }
// }

template <typename T>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void alapgrad_gsrb_with_line_solve(
    Box const & /*box*/, Array4<T> const & /*phi*/,
    Array4<T const> const & /*rhs*/, T /*alpha*/,
    Array4<T const> const & /*a*/, T /*dhx*/, Array4<T const> const & /*bX*/,
    Array4<int const> const & /*m0*/, Array4<int const> const & /*m1*/,
    Array4<T const> const & /*f0*/, Array4<T const> const & /*f1*/,
    Box const & /*vbox*/, int /*redblack*/, int /*nc*/) noexcept
{
    amrex::Abort("alapgrad_gsrb_with_line_solve not implemented in 1D");
}

// template <typename T>
// AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
// void overset_rescale_bcoef_x (Box const& box, Array4<T> const& bX,
// Array4<int const> const& osm,
//                               int ncomp, T osfac) noexcept
// {
//     const auto lo = amrex::lbound(box);
//     const auto hi = amrex::ubound(box);
//     for (int n = 0; n < ncomp; ++n) {
//         for (int i = lo.x; i <= hi.x; ++i) {
//             if ((osm(i-1,0,0)+osm(i,0,0)) == 1) {
//                 bX(i,0,0,n) *= osfac;
//             }
//         }
//     }
// }

}
#endif
