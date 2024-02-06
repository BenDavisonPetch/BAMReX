#include "RiemannSolver.H"

using namespace amrex;

AMREX_GPU_HOST
void compute_exact_RP_solution(
    const int dir, const amrex::Real time, const amrex::Box &bx,
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> &dx_arr,
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> &prob_lo,
    const amrex::Real                                   initial_position,
    const amrex::GpuArray<amrex::Real, MAX_STATE_SIZE> &primv_L,
    const amrex::GpuArray<amrex::Real, MAX_STATE_SIZE> &primv_R,
    const amrex::Real adiabatic_L, const amrex::Real adiabatic_R,
    const amrex::Array4<amrex::Real> &dest)
{
    // speeds
    const Real u_L = primv_L[1 + dir];
    const Real u_R = primv_R[1 + dir];
    const Real c_L = sound_speed(primv_L, adiabatic_L);
    const Real c_R = sound_speed(primv_R, adiabatic_R);
    // pressures
    const Real p_L = primv_L[1 + AMREX_SPACEDIM];
    const Real p_R = primv_R[1 + AMREX_SPACEDIM];
    // calculate intermediate pressure
    const Real p_interm
        = euler_RP_interm_p(dir, primv_L, primv_R, adiabatic_L, adiabatic_R);
    const Real u_interm = 0.5 * (u_L + u_R)
                          + 0.5
                                * (euler_riemann_function_one_side(
                                       dir, p_interm, primv_R, adiabatic_R)
                                   - euler_riemann_function_one_side(
                                       dir, p_interm, primv_L, adiabatic_L));

    // primitive values for intermediate states
    GpuArray<Real, MAX_STATE_SIZE> primv_interm_L, primv_interm_R;
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
    Real speed_L_head, speed_L_tail;
    Real speed_R_head, speed_R_tail;
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

    const auto consv_L        = consv_from_primv(primv_L, adiabatic_L);
    const auto consv_R        = consv_from_primv(primv_R, adiabatic_R);
    const auto consv_interm_L = consv_from_primv(primv_interm_L, adiabatic_L);
    const auto consv_interm_R = consv_from_primv(primv_interm_R, adiabatic_R);

    AMREX_ASSERT(speed_L_head <= speed_L_tail);
    AMREX_ASSERT(speed_L_tail <= u_interm);
    AMREX_ASSERT(u_interm <= speed_R_tail);
    AMREX_ASSERT(speed_R_tail <= speed_R_head);

    const unsigned int NSTATE = AmrLevelAdv::NUM_STATE;
    // Loop over cells
    ParallelFor(
        bx,
        [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            // select correct index corresponding to the direction
            const int dir_idx = (dir == 0) ? i : ((dir == 1) ? j : k);
            // calculate position of current grid cell
            const Real pos = prob_lo[dir] + (dir_idx + 0.5) * dx_arr[dir];
            const Real S   = (pos - initial_position) / time;

            if (S <= speed_L_head)
            {
                // left of left-moving wave
                for (int n = 0; n < NSTATE; ++n) dest(i, j, k, n) = consv_L[n];
            }
            else if (S <= speed_L_tail)
            {
                // inside fan of left-moving rarefaction
                GpuArray<Real, MAX_STATE_SIZE> primv_fan;
                AMREX_D_TERM(primv_fan[1] = primv_L[1];
                             , primv_fan[2] = primv_L[2];
                             , primv_fan[3] = primv_L[3];)
                const Real tmp = 2 / (adiabatic_L + 1)
                                 + (adiabatic_L - 1)
                                       / ((adiabatic_L + 1) * c_L) * (u_L - S);
                primv_fan[0]
                    = primv_L[0] * std::pow(tmp, 2 / (adiabatic_L - 1));
                primv_fan[1 + dir]
                    = 2 / (adiabatic_L + 1)
                      * (c_L + (adiabatic_L - 1) * u_L * 0.5 + S);
                primv_fan[1 + AMREX_SPACEDIM]
                    = p_L * std::pow(tmp, 2 * adiabatic_L / (adiabatic_L - 1));

                const auto consv_fan
                    = consv_from_primv(primv_fan, adiabatic_L);

                for (int n = 0; n < NSTATE; ++n)
                    dest(i, j, k, n) = consv_fan[n];
            }
            else if (S <= u_interm)
            {
                // between left-moving wave and contact wave
                for (int n = 0; n < NSTATE; ++n)
                    dest(i, j, k, n) = consv_interm_L[n];
            }
            else if (S <= speed_R_tail)
            {
                // between right moving wave and contact wave
                for (int n = 0; n < NSTATE; ++n)
                    dest(i, j, k, n) = consv_interm_R[n];
            }
            else if (S <= speed_R_head)
            {
                // inside fan of right moving rarefaction
                GpuArray<Real, MAX_STATE_SIZE> primv_fan;
                AMREX_D_TERM(primv_fan[1] = primv_R[1];
                             , primv_fan[2] = primv_R[2];
                             , primv_fan[3] = primv_R[3];)
                const Real tmp = 2 / (adiabatic_R + 1)
                                 - (adiabatic_R - 1)
                                       / ((adiabatic_R + 1) * c_R) * (u_R - S);
                primv_fan[0]
                    = primv_R[0] * std::pow(tmp, 2 / (adiabatic_R - 1));
                primv_fan[1 + dir]
                    = 2 / (adiabatic_R + 1)
                      * (-c_R + (adiabatic_R - 1) * u_R * 0.5 + S);
                primv_fan[1 + AMREX_SPACEDIM]
                    = p_R * std::pow(tmp, 2 * adiabatic_R / (adiabatic_R - 1));

                const auto consv_fan
                    = consv_from_primv(primv_fan, adiabatic_R);

                for (int n = 0; n < NSTATE; ++n)
                    dest(i, j, k, n) = consv_fan[n];
            }
            else
            {
                // right of right-moving wave
                for (int n = 0; n < NSTATE; ++n) dest(i, j, k, n) = consv_R[n];
            }
        });
}