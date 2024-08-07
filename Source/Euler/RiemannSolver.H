#ifndef RiemannSolver_H_
#define RiemannSolver_H_

#include "Euler/Euler.H"
#include "RootFinding.H"
#include <AMReX_Extension.H>
#include <AMReX_GpuQualifiers.H>

template <int orientation, int dim = AMREX_SPACEDIM>
class EulerExactRiemannSolver
{
  public:
    static constexpr int NCOMP = dim + 2;

    EulerExactRiemannSolver(const amrex::GpuArray<amrex::Real, NCOMP> &primv_L,
                            const amrex::GpuArray<amrex::Real, NCOMP> &primv_R,
                            const amrex::Real adiabatic_L,
                            const amrex::Real adiabatic_R,
                            const amrex::Real epsilon);

    //! You must call this before you try sampling or getting intermediate
    //! pressure
    AMREX_GPU_HOST_DEVICE
    AMREX_FORCE_INLINE
    void compute_soln();

    //! Fills @c dst with the solution at S = x/t for an interface at x = 0
    AMREX_GPU_HOST_DEVICE
    AMREX_FORCE_INLINE
    void sample(const amrex::Real S, const amrex::Array4<amrex::Real> &dst,
                int i, int j, int k);

    AMREX_GPU_HOST
    void fill(const amrex::Real time, const amrex::Real interface_pos,
              const amrex::Box                        &bx,
              const amrex::GpuArray<amrex::Real, dim> &dx_arr,
              const amrex::GpuArray<amrex::Real, dim> &prob_lo,
              const amrex::Array4<amrex::Real>        &dst);

    //! Get pressure of intermediate state
    AMREX_GPU_HOST_DEVICE
    AMREX_FORCE_INLINE
    amrex::Real get_p_interm()
    {
        if (!p_interm_computed)
            compute_p_interm();
        return p_interm;
    }

    AMREX_GPU_HOST_DEVICE
    AMREX_FORCE_INLINE
    const amrex::GpuArray<amrex::Real, NCOMP> &get_interm_consv_L()
    {
        if (!interm_states_computed)
        {
            compute_soln();
        }
        return consv_interm_L;
    }

    AMREX_GPU_HOST_DEVICE
    AMREX_FORCE_INLINE
    const amrex::GpuArray<amrex::Real, NCOMP> &get_interm_consv_R()
    {
        if (!interm_states_computed)
        {
            compute_soln();
        }
        return consv_interm_R;
    }

  protected:
    /**
     * @brief Uses the Newton-Raphson root finding method to find the
     * intermediate pressure for the Euler equation Riemann problem with two
     * ideal gases. Uses the initial guess from a linearised solution based on
     * primitive variables discussed in Toro's book Riemann Solvers and
     * Numerical Methods for Fluid Dynamics
     *
     * @param dir The direction the interface is normal to
     * @param primv_L The primitive values of the left state
     * @param primv_R The primitive values of the right state
     * @param adiabatic_L The adiabatic constant of the left state
     * @param adiabatic_R The adiabatic constant of the right state
     */
    AMREX_GPU_HOST_DEVICE
    AMREX_FORCE_INLINE
    void compute_p_interm();

    const amrex::GpuArray<amrex::Real, NCOMP> &primv_L;
    const amrex::GpuArray<amrex::Real, NCOMP> &primv_R;
    const amrex::GpuArray<amrex::Real, NCOMP>  consv_L, consv_R;
    const amrex::Real                          adiabatic_L, adiabatic_R;
    const amrex::Real                          epsilon;

    bool        p_interm_computed = false;
    amrex::Real p_interm          = 0;

    bool                                interm_states_computed = false;
    amrex::GpuArray<amrex::Real, NCOMP> consv_interm_L, consv_interm_R;

    bool        wave_speeds_computed = false;
    amrex::Real u_interm;
    amrex::Real speed_L_head, speed_L_tail, speed_R_head, speed_R_tail;
};

//
// Implementations
//
template <int orientation, int dim>
EulerExactRiemannSolver<orientation, dim>::EulerExactRiemannSolver(
    const amrex::GpuArray<amrex::Real, NCOMP> &primv_L,
    const amrex::GpuArray<amrex::Real, NCOMP> &primv_R,
    const amrex::Real adiabatic_L, const amrex::Real adiabatic_R,
    const amrex::Real epsilon)
    : primv_L(primv_L)
    , primv_R(primv_R)
    , consv_L(consv_from_primv(primv_L, adiabatic_L, epsilon))
    , consv_R(consv_from_primv(primv_R, adiabatic_R, epsilon))
    , adiabatic_L(adiabatic_L)
    , adiabatic_R(adiabatic_R)
    , epsilon(epsilon)
{
}

template <int orientation, int dim>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
EulerExactRiemannSolver<orientation, dim>::compute_p_interm()
{
    // Calculate initial guess for pressure
    amrex::Real c_L = sound_speed(primv_L, adiabatic_L, 1);
    amrex::Real c_R = sound_speed(primv_R, adiabatic_R, 1);
    p_interm
        = 0.5 * (primv_L[1 + AMREX_SPACEDIM] + primv_R[1 + AMREX_SPACEDIM])
          - 0.125 * sqrt(epsilon)
                * (primv_R[1 + orientation] - primv_L[1 + orientation])
                * (primv_L[0] + primv_R[0]) * (c_L + c_R);
    amrex::Real TOL = 1e-6;
    p_interm        = std::max(TOL, p_interm);

    // Use NR method to find root
    auto f = [=] AMREX_GPU_HOST_DEVICE(amrex::Real p)
    {
        return euler_riemann_function<orientation>(
            p, primv_L, primv_R, adiabatic_L, adiabatic_R, epsilon);
    };
    auto f_deriv = [=] AMREX_GPU_HOST_DEVICE(amrex::Real p)
    {
        return euler_riemann_function_deriv(p, primv_L, primv_R, adiabatic_L,
                                            adiabatic_R);
    };
    newton_raphson(p_interm, f, f_deriv);
    AMREX_ASSERT(p_interm == p_interm);
    p_interm_computed = true;
}

template <int dir, int dim>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
EulerExactRiemannSolver<dir, dim>::compute_soln()
{
    using namespace amrex;
    // speeds
    const Real u_L = primv_L[1 + dir];
    const Real u_R = primv_R[1 + dir];
    const Real c_L = sound_speed(primv_L, adiabatic_L, epsilon);
    const Real c_R = sound_speed(primv_R, adiabatic_R, epsilon);
    // pressures
    const Real p_L = primv_L[1 + AMREX_SPACEDIM];
    const Real p_R = primv_R[1 + AMREX_SPACEDIM];
    // calculate intermediate pressure
    if (!p_interm_computed)
        compute_p_interm();

    u_interm = 0.5 * (u_L + u_R)
               + 0.5
                     * (euler_riemann_function_one_side(p_interm, primv_R,
                                                        adiabatic_R)
                        - euler_riemann_function_one_side(p_interm, primv_L,
                                                          adiabatic_L))
                     / sqrt(epsilon);

    // primitive values for intermediate states
    GpuArray<Real, EULER_NCOMP> primv_interm_L, primv_interm_R;
    // copy over velocities
    AMREX_D_TERM(primv_interm_L[1] = primv_L[1];
                 , primv_interm_L[2] = primv_L[2];
                 , primv_interm_L[3] = primv_L[3];)
    AMREX_D_TERM(primv_interm_R[1] = primv_R[1];
                 , primv_interm_R[2] = primv_R[2];
                 , primv_interm_R[3] = primv_R[3];)
    // update velocities in direction dir
    primv_interm_L[1 + dir] = u_interm;
    primv_interm_R[1 + dir] = u_interm;
    // update pressures
    primv_interm_L[1 + AMREX_SPACEDIM] = p_interm;
    primv_interm_R[1 + AMREX_SPACEDIM] = p_interm;

    // The speeds of the head and tail of the left and right moving waves. If a
    // wave is a shock, the tail speed is equal to the head speed.
    if (p_interm > p_L)
    {
        // Left moving shock

        // density
        const Real p_ratio  = p_interm / p_L;
        const Real ad_ratio = (adiabatic_L - 1) / (adiabatic_L + 1);
        primv_interm_L[0]
            = primv_L[0] * (p_ratio + ad_ratio) / (ad_ratio * p_ratio + 1);

        // wave speeds
        speed_L_head
            = u_L
              - c_L
                    * std::sqrt(((adiabatic_L + 1) * p_interm)
                                    / (2 * adiabatic_L * p_L)
                                + (adiabatic_L - 1) / (2 * adiabatic_L));
        speed_L_tail = speed_L_head;
    }
    else
    {
        // Left moving rarefaction

        // density
        primv_interm_L[0]
            = primv_L[0] * std::pow(p_interm / p_L, 1 / adiabatic_L);
        // sound speed behind rarefaction
        const Real c_interm_L
            = c_L
              * std::pow(p_interm / p_L,
                         (adiabatic_L - 1) / (2 * adiabatic_L));

        // wave speeds
        speed_L_head = u_L - c_L;
        speed_L_tail = u_interm - c_interm_L;
    }

    if (p_interm > p_R)
    {
        // Right moving shock

        // density
        const Real p_ratio  = p_interm / p_R;
        const Real ad_ratio = (adiabatic_R - 1) / (adiabatic_R + 1);
        primv_interm_R[0]
            = primv_R[0] * (p_ratio + ad_ratio) / (ad_ratio * p_ratio + 1);

        // wave speeds
        speed_R_head
            = u_R
              + c_R
                    * std::sqrt(((adiabatic_R + 1) * p_interm)
                                    / (2 * adiabatic_R * p_R)
                                + (adiabatic_R - 1) / (2 * adiabatic_R));
        speed_R_tail = speed_R_head;
    }
    else
    {
        // Right moving rarefaction

        // density
        primv_interm_R[0]
            = primv_R[0] * std::pow(p_interm / p_R, 1 / adiabatic_R);
        // sound speed behind rarefaction
        const Real c_interm_R
            = c_R
              * std::pow(p_interm / p_R,
                         (adiabatic_R - 1) / (2 * adiabatic_R));

        // wave speeds
        speed_R_head = u_R + c_R;
        speed_R_tail = u_interm + c_interm_R;
    }

    consv_interm_L = consv_from_primv(primv_interm_L, adiabatic_L, epsilon);
    consv_interm_R = consv_from_primv(primv_interm_R, adiabatic_R, epsilon);

    AMREX_ASSERT(speed_L_head <= speed_L_tail + 1e-12 * fabs(speed_L_tail));
    AMREX_ASSERT(speed_L_tail <= u_interm);
    AMREX_ASSERT(u_interm <= speed_R_tail);
    AMREX_ASSERT(speed_R_tail <= speed_R_head + 1e-12 * fabs(speed_R_head));

    interm_states_computed = true;
    wave_speeds_computed   = true;
}

template <int dir, int dim>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
EulerExactRiemannSolver<dir, dim>::sample(
    const amrex::Real S, const amrex::Array4<amrex::Real> &dst, int i, int j,
    int k)
{
    using namespace amrex;

    AMREX_ASSERT(p_interm_computed);
    AMREX_ASSERT(interm_states_computed);
    AMREX_ASSERT(wave_speeds_computed);

    // For ease of reading
    const Real u_L = primv_L[1 + dir];
    const Real u_R = primv_R[1 + dir];
    const Real c_L = sound_speed(primv_L, adiabatic_L, epsilon);
    const Real c_R = sound_speed(primv_R, adiabatic_R, epsilon);
    // pressures
    const Real p_L = primv_L[1 + AMREX_SPACEDIM];
    const Real p_R = primv_R[1 + AMREX_SPACEDIM];

    if (S <= speed_L_head)
    {
        // left of left-moving wave
        for (int n = 0; n < NCOMP; ++n) dst(i, j, k, n) = consv_L[n];
    }
    else if (S <= speed_L_tail)
    {
        // inside fan of left-moving rarefaction
        GpuArray<Real, EULER_NCOMP> primv_fan;
        AMREX_D_TERM(primv_fan[1] = primv_L[1];, primv_fan[2] = primv_L[2];
                     , primv_fan[3]                           = primv_L[3];)
        const Real tmp
            = 2 / (adiabatic_L + 1)
              + (adiabatic_L - 1) / ((adiabatic_L + 1) * c_L) * (u_L - S);
        primv_fan[0]       = primv_L[0] * std::pow(tmp, 2 / (adiabatic_L - 1));
        primv_fan[1 + dir] = 2 / (adiabatic_L + 1)
                             * (c_L + (adiabatic_L - 1) * u_L * 0.5 + S);
        primv_fan[1 + AMREX_SPACEDIM]
            = p_L * std::pow(tmp, 2 * adiabatic_L / (adiabatic_L - 1));

        const auto consv_fan
            = consv_from_primv(primv_fan, adiabatic_L, epsilon);

        for (int n = 0; n < NCOMP; ++n) dst(i, j, k, n) = consv_fan[n];
    }
    else if (S <= u_interm)
    {
        // between left-moving wave and contact wave
        for (int n = 0; n < NCOMP; ++n) dst(i, j, k, n) = consv_interm_L[n];
    }
    else if (S <= speed_R_tail)
    {
        // between right moving wave and contact wave
        for (int n = 0; n < NCOMP; ++n) dst(i, j, k, n) = consv_interm_R[n];
    }
    else if (S <= speed_R_head)
    {
        // inside fan of right moving rarefaction
        GpuArray<Real, EULER_NCOMP> primv_fan;
        AMREX_D_TERM(primv_fan[1] = primv_R[1];, primv_fan[2] = primv_R[2];
                     , primv_fan[3]                           = primv_R[3];)
        const Real tmp
            = 2 / (adiabatic_R + 1)
              - (adiabatic_R - 1) / ((adiabatic_R + 1) * c_R) * (u_R - S);
        primv_fan[0]       = primv_R[0] * std::pow(tmp, 2 / (adiabatic_R - 1));
        primv_fan[1 + dir] = 2 / (adiabatic_R + 1)
                             * (-c_R + (adiabatic_R - 1) * u_R * 0.5 + S);
        primv_fan[1 + AMREX_SPACEDIM]
            = p_R * std::pow(tmp, 2 * adiabatic_R / (adiabatic_R - 1));

        const auto consv_fan
            = consv_from_primv(primv_fan, adiabatic_R, epsilon);

        for (int n = 0; n < NCOMP; ++n) dst(i, j, k, n) = consv_fan[n];
    }
    else
    {
        // right of right-moving wave
        for (int n = 0; n < NCOMP; ++n) dst(i, j, k, n) = consv_R[n];
    }
}

template <int dir, int dim>
AMREX_GPU_HOST void EulerExactRiemannSolver<dir, dim>::fill(
    const amrex::Real time, const amrex::Real interface_pos,
    const amrex::Box &bx, const amrex::GpuArray<amrex::Real, dim> &dx_arr,
    const amrex::GpuArray<amrex::Real, dim> &prob_lo,
    const amrex::Array4<amrex::Real>        &dst)
{
    using namespace amrex;
    if (!interm_states_computed || !wave_speeds_computed)
        compute_soln();

    ParallelFor(bx,
                [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    // select correct index corresponding to the direction
                    const int dir_idx = (dir == 0) ? i : ((dir == 1) ? j : k);
                    // calculate position of current grid cell
                    const Real pos
                        = prob_lo[dir] + (dir_idx + 0.5) * dx_arr[dir];
                    const Real S = (pos - interface_pos) / time;
                    sample(S, dst, i, j, k);
                });
}

#endif