#ifndef HLLC_H_
#define HLLC_H_

#include "AmrLevelAdv.H"
#include "Euler/Euler.H"
#include "Exceptions.H"
#include <AMReX_Array4.H>
#include <AMReX_Box.H>
#include <AMReX_Extension.H>
#include <AMReX_GpuQualifiers.H>

/*
 * Note that the function declaration for computing the HLLC flux can be found
 * in Fluxes.H. This file contains functions within the HLLC solver.
 * Both the computation of the flux and the functions declared here are
 * implemented in HLLC.cpp
 */

/**
 * @brief Estimates the wave speed estimates for use in the HLLC Riemann
 * solver, following approach TMSb of Toro et al. (2020):
 * https://arxiv.org/pdf/2005.04898.pdf
 * Assumes ideal gas EoS for the Euler equations
 *
 * \param dir direction of flux being calculated
 * \param i x index of flux being calculated
 * \param j y index of flux being calculated
 * \param k z index of flux being calculated
 * \param consv_values_L
 * \param consv_values_R
 * \param speed_L returns the estimate for the left wave speed
 * \param speed_R returns the estimate for the right wave speed
 * \param speed_interm returns the estimate for the intermediate wave speed
 *
 * @details The variables used as the left state are taken from
 * \c consv_values_L and the variables used as the right state are taken from
 * \c consv_values_R. The indexes of the right state are (i,j,k) and the
 * indexes of the left state are decreased by 1 in direction \c dir
 */
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void estimate_wave_speed_HLLC_TMSb(
    const int dir, const int i, const int j, const int k,
    const amrex::Array4<const amrex::Real> &consv_values_L,
    const amrex::Array4<const amrex::Real> &consv_values_R,
    amrex::Real adiabatic_L, amrex::Real adiabatic_R, amrex::Real &speed_L,
    amrex::Real &speed_R, amrex::Real &speed_interm);

/**
 * @brief Estimates the wave speed estimates for use in the HLLC Riemann
 * solver, following the 1994 approach of Toro
 *
 * \param dir direction of flux being calculated
 * \param i x index of flux being calculated
 * \param j y index of flux being calculated
 * \param k z index of flux being calculated
 * \param consv_values_L
 * \param consv_values_R
 * \param speed_L returns the estimate for the left wave speed
 * \param speed_R returns the estimate for the right wave speed
 * \param speed_interm returns the estimate for the intermediate wave speed
 *
 * @details The variables used as the left state are taken from
 * \c consv_values_L and the variables used as the right state are taken from
 * \c consv_values_R. The indexes of the right state are (i,j,k) and the
 * indexes of the left state are decreased by 1 in direction \c dir
 */
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void estimate_wave_speed_HLLC_Toro(
    int dir, int i, int j, int k,
    const amrex::Array4<const amrex::Real> &consv_values_L,
    const amrex::Array4<const amrex::Real> &consv_values_R,
    amrex::Real adiabatic_L, amrex::Real adiabatic_R, amrex::Real &speed_L,
    amrex::Real &speed_R, amrex::Real &speed_interm);

/**
 * @brief Replaces the array \c primv with the intermediate state (in
 * conservative form) in one side of an HLLC approximate Riemann solver for
 * the Euler equations.
 *
 * @param dir The direction in which the flux is being calculated.
 * @param primv The primitive variables of the state on side K (L or R).
 * Overwritten by result.
 * @param energy The volumetric internal energy of the state on side K (L
 * or R)
 * @param speed_K The wave speed estimate for the K direction moving wave
 * @param speed_interm The wave seped estimate for the contact wave.
 */
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void HLLC_interm_state_replace(
    const int dir, amrex::GpuArray<amrex::Real, EULER_NCOMP> &primv,
    amrex::Real energy, amrex::Real speed_K, amrex::Real speed_interm);

//
// Implementations
//
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void estimate_wave_speed_HLLC_TMSb(
    const int dir, const int i, const int j, const int k,
    const amrex::Array4<const amrex::Real> &consv_values_L,
    const amrex::Array4<const amrex::Real> &consv_values_R,
    amrex::Real adiabatic_L, amrex::Real adiabatic_R, amrex::Real &speed_L,
    amrex::Real &speed_R, amrex::Real &speed_interm)
{
    const int i_off = (dir == 0) ? 1 : 0; // TODO template direction
    const int j_off = (dir == 1) ? 1 : 0;
    const int k_off = (dir == 2) ? 1 : 0;

    const auto primv_L = primv_from_consv(consv_values_L, adiabatic_L,
                                          i - i_off, j - j_off, k - k_off);
    const auto primv_R
        = primv_from_consv(consv_values_R, adiabatic_R, i, j, k);

    const amrex::Real density_L  = primv_L[0];
    const amrex::Real density_R  = primv_R[0];
    const amrex::Real pressure_L = primv_L[1 + AMREX_SPACEDIM];
    const amrex::Real pressure_R = primv_R[1 + AMREX_SPACEDIM];
    const amrex::Real u_L        = primv_L[1 + dir];
    const amrex::Real u_R        = primv_R[1 + dir];
    // sound speeds
    const amrex::Real c_L = sound_speed(primv_L, adiabatic_L);
    const amrex::Real c_R = sound_speed(primv_R, adiabatic_R);

    AMREX_ASSERT_WITH_MESSAGE(adiabatic_L == adiabatic_R, "Not implemented!");
    const amrex::Real adiabatic = adiabatic_L;

    const amrex::Real all_raref_pressure = std::pow(
        (c_L + c_R - 0.5 * (adiabatic - 1) * (u_R - u_L))
            / (c_L / std::pow(pressure_L, (adiabatic - 1) / (2 * adiabatic))
               + c_R
                     / std::pow(pressure_R,
                                (adiabatic - 1) / (2 * adiabatic))),
        2 * adiabatic / (adiabatic - 1));
    const amrex::Real p_min = std::min(pressure_L, pressure_R);
    const amrex::Real p_max = std::max(pressure_L, pressure_R);

    // riemann functions evaluated at p_min and p_max
    const amrex::Real f_p_min = euler_riemann_function(
        dir, p_min, primv_L, primv_R, adiabatic_L, adiabatic_R);
    const amrex::Real f_p_max = euler_riemann_function(
        dir, p_max, primv_L, primv_R, adiabatic_L, adiabatic_R);
    if (f_p_min >= 0)
    {
        speed_L = u_L - c_L;
        speed_R = u_R + c_R;
    }
    else if (f_p_min < 0 && f_p_max > 0 && p_min == pressure_R)
    {
        // following notation in paper
        const amrex::Real f_p_allraref
            = euler_riemann_function(dir, all_raref_pressure, primv_L, primv_R,
                                     adiabatic_L, adiabatic_R);
        const amrex::Real p_mrr = p_min
                                  - (all_raref_pressure - p_min)
                                        / (f_p_allraref - f_p_min) * f_p_min;
        const amrex::Real q_R
            = std::sqrt(1
                        + (adiabatic + 1) / (2 * adiabatic)
                              * (p_mrr / pressure_R
                                 - 1)); // bottom of P11 of paper linked at top

        speed_L = u_L - c_L;
        speed_R = u_R + c_R * q_R;
    }
    else if (f_p_min < 0 && f_p_max > 0 && p_min == pressure_L)
    {
        // following notation in paper
        const amrex::Real f_p_allraref
            = euler_riemann_function(dir, all_raref_pressure, primv_L, primv_R,
                                     adiabatic_L, adiabatic_R);
        const amrex::Real p_mrr = p_min
                                  - (all_raref_pressure - p_min)
                                        / (f_p_allraref - f_p_min) * f_p_min;
        const amrex::Real q_L = std::sqrt(
            1 + (adiabatic + 1) / (2 * adiabatic) * (p_mrr / pressure_L - 1));

        speed_R = u_R + c_R;
        speed_L = u_L - c_L * q_L;
    }
    else if (f_p_max < 0)
    {
        const amrex::Real f_p_allraref
            = euler_riemann_function(dir, all_raref_pressure, primv_L, primv_R,
                                     adiabatic_L, adiabatic_R);
        const amrex::Real p_mrr = p_min
                                  - (all_raref_pressure - p_min)
                                        / (f_p_allraref - f_p_min) * f_p_min;
        const amrex::Real q_R
            = std::sqrt(1
                        + (adiabatic + 1) / (2 * adiabatic)
                              * (p_mrr / pressure_R
                                 - 1)); // bottom of P11 of paper linked at top
        const amrex::Real q_L = std::sqrt(
            1 + (adiabatic + 1) / (2 * adiabatic) * (p_mrr / pressure_L - 1));

        speed_R = u_R + c_R * q_R;
        speed_L = u_L - c_L * q_L;
    }
    else
    {
        AMREX_ASSERT(false); // internal error
    }

    speed_interm
        = (pressure_R - pressure_L + density_L * u_L * (speed_L - u_L)
           - density_R * u_R * (speed_R - u_R))
          / (density_L * (speed_L - u_L) - density_R * (speed_R - u_R));
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void estimate_wave_speed_HLLC_Toro(
    int dir, int i, int j, int k,
    const amrex::Array4<const amrex::Real> &consv_values_L,
    const amrex::Array4<const amrex::Real> &consv_values_R,
    amrex::Real adiabatic_L, amrex::Real adiabatic_R, amrex::Real &speed_L,
    amrex::Real &speed_R, amrex::Real &speed_interm)
{
    const int i_off = (dir == 0) ? 1 : 0; // TODO template direction
    const int j_off = (dir == 1) ? 1 : 0;
    const int k_off = (dir == 2) ? 1 : 0;

    auto primv_L = primv_from_consv(consv_values_L, adiabatic_L, i - i_off,
                                    j - j_off, k - k_off);
    auto primv_R = primv_from_consv(consv_values_R, adiabatic_R, i, j, k);

    const amrex::Real density_L = primv_L[0];
    const amrex::Real density_R = primv_R[0];
    const amrex::Real p_L       = primv_L[1 + AMREX_SPACEDIM];
    const amrex::Real p_R       = primv_R[1 + AMREX_SPACEDIM];
    const amrex::Real u_L       = primv_L[1 + dir];
    const amrex::Real u_R       = primv_R[1 + dir];
    // sound speeds
    const amrex::Real c_L = sound_speed(primv_L, adiabatic_L);
    const amrex::Real c_R = sound_speed(primv_R, adiabatic_R);

    AMREX_ASSERT_WITH_MESSAGE(
        density_L > 0,
        "Non-positive density! Consider changing the limiting strategy");
    AMREX_ASSERT_WITH_MESSAGE(
        density_R > 0,
        "Non-positive density! Consider changing the limiting strategy");
    AMREX_ASSERT_WITH_MESSAGE(
        p_L > 0,
        "Non-positive pressure! Consider changing the limiting strategy");
    AMREX_ASSERT_WITH_MESSAGE(
        p_R > 0,
        "Non-positive pressure! Consider changing the limiting strategy");
    AMREX_ASSERT_WITH_MESSAGE(adiabatic_L == adiabatic_R, "Not implemented!");
    const amrex::Real adiabatic = adiabatic_L;

    const amrex::Real z = (adiabatic - 1) / (2 * adiabatic);
    const amrex::Real p_rr
        = std::pow((c_L + c_R - (adiabatic - 1) / 2 * (u_R - u_L))
                       / (c_L * std::pow(p_L, -z) + c_R * std::pow(p_R, -z)),
                   1 / z);
    const amrex::Real q_L
        = (p_rr > p_L) ? std::sqrt(
              1 + (adiabatic + 1) / (2 * adiabatic) * (p_rr / p_L - 1))
                       : 1;
    const amrex::Real q_R
        = (p_rr > p_R) ? std::sqrt(
              1 + (adiabatic + 1) / (2 * adiabatic) * (p_rr / p_R - 1))
                       : 1;

    speed_L = u_L - c_L * q_L;
    speed_R = u_R + c_R * q_R;

    const amrex::Real denom
        = (density_L * (speed_L - u_L) - density_R * (speed_R - u_R));
    speed_interm = (denom != 0)
                       ? (p_R - p_L + density_L * u_L * (speed_L - u_L)
                          - density_R * u_R * (speed_R - u_R))
                             / denom
                       : 0;

    // if (speed_L != speed_L || speed_R != speed_R || speed_interm !=
    // speed_interm)
    // {
    //     std::cout << "(i, j, k) = " << i << " " << j << " " << k <<
    //     std::endl; std::cout << "d_L = " << density_L << "\tp_L = " << p_L
    //             << "\tu_L = " << u_L << std::endl;
    //     std::cout << "d_R = " << density_L << "\tp_R = " << p_R
    //             << "\tu_R = " << u_R << std::endl;
    //     std::cout << "c_L = " << c_L << "\tc_R = " << c_R << std::endl;
    //     std::cout << "S_L = " << speed_L << "\tS_interm = " << speed_interm
    //             << "\tS_R = " << speed_R << std::endl;
    // }
    ASSERT_NOT_NAN(speed_L);
    ASSERT_NOT_NAN(speed_R);
    ASSERT_NOT_NAN(speed_interm);
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void HLLC_interm_state_replace(
    const int dir, amrex::GpuArray<amrex::Real, EULER_NCOMP> &primv,
    amrex::Real energy, amrex::Real speed_K, amrex::Real speed_interm)
{
    const amrex::Real density  = primv[0];
    const amrex::Real u_K      = primv[1 + dir];
    const amrex::Real pressure = primv[1 + AMREX_SPACEDIM];

    primv[0]       = 1;
    primv[1 + dir] = speed_interm;

    const amrex::Real scale_factor
        = density * (speed_K - u_K) / (speed_K - speed_interm);

    primv[1 + AMREX_SPACEDIM]
        = energy / density
          + (speed_interm - u_K)
                * (speed_interm + pressure / (density * (speed_K - u_K)));

    for (int i = 0; i < EULER_NCOMP; ++i)
    {
        primv[i] *= scale_factor;
    }
}

#endif