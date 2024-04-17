#include <AMReX_GpuContainers.H>
#include <AMReX_ParmParse.H>
#include <AMReX_REAL.H>

#include "AmrLevelAdv.H"
#include "Fluxes/Limiters.H"
#include "GPU/CopyToSymbol.H"

AMREX_GPU_CONSTANT ProbParm d_prob_parm;

unsigned int var_name_to_idx(const std::string &var_name)
{
    if (var_name == "density")
        return 0;
    else if (var_name == "mom_x")
        return 1;
    else if (var_name == "mom_y")
        return 2;
    else if (var_name == "mom_z")
        return 3;
    else if (var_name == "energy")
        return 1 + AMREX_SPACEDIM;
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(false,
                                     "Invalid variable name " + var_name);
    return 0;
}

AMREX_GPU_HOST
void set_h_prob_parm_defaults()
{
    AmrLevelAdv::h_prob_parm->adiabatic = -1; // must set adiabatic
    AmrLevelAdv::h_prob_parm->epsilon   = 1;
    AmrLevelAdv::h_prob_parm->limiter   = van_leer_limiter;
    for (int i = 0; i < EULER_NCOMP; ++i)
        AmrLevelAdv::h_prob_parm->limiter_indices[i] = 1 + AMREX_SPACEDIM;
}

extern "C"
{
    void amrex_probinit(const int * /*init*/, const int * /*name*/,
                        const int * /*namelen*/,
                        const amrex::Real * /*problo*/,
                        const amrex::Real * /*probhi*/)
    {
        set_h_prob_parm_defaults();
        {
            // Read the prob block from the input file using ParmParse
            amrex::ParmParse pp("prob");
            amrex::Real      adiabatic = 1.4;
            amrex::Real      epsilon   = 1;
            pp.get("adiabatic", adiabatic);
            pp.query("epsilon", epsilon);
            AmrLevelAdv::h_prob_parm->adiabatic = adiabatic;
            AmrLevelAdv::h_prob_parm->epsilon   = epsilon;
        }
        {
            // Read the limiter block from the input file
            amrex::ParmParse         pp("limiting");
            std::string              limiter_name;
            std::string              density_limiter, energy_limiter;
            std::vector<std::string> mom_limiters(3);
            pp.query("limiter", limiter_name);
            pp.query("density", density_limiter);
            pp.queryarr("mom", mom_limiters);
            pp.query("energy", energy_limiter);
            if (!limiter_name.empty())
                AmrLevelAdv::h_prob_parm->limiter
                    = limiter_from_name(limiter_name);

            if (!density_limiter.empty())
                AmrLevelAdv::h_prob_parm->limiter_indices[0]
                    = var_name_to_idx(density_limiter);
            if (!energy_limiter.empty())
                AmrLevelAdv::h_prob_parm->limiter_indices[1 + AMREX_SPACEDIM]
                    = var_name_to_idx(energy_limiter);
            for (int i = 0; i < AMREX_SPACEDIM; ++i)
                if (!mom_limiters[i].empty())
                    AmrLevelAdv::h_prob_parm->limiter_indices[1 + i]
                        = var_name_to_idx(mom_limiters[i]);
        }

        // Transfer the problem-specific data to the GPU
        // amrex::Gpu::copy(amrex::Gpu::hostToDevice, AmrLevelAdv::h_prob_parm,
        //                  AmrLevelAdv::h_prob_parm + 1,
        //                  AmrLevelAdv::d_prob_parm);
        bamrex::gpu::copy_to_symbol(&d_prob_parm, AmrLevelAdv::h_prob_parm,
                                    AmrLevelAdv::h_prob_parm + 1);
    }
}
