#include "IMEXSettings.H"
#include "IMEX/ButcherTableau.H"

#include <AMReX_ParmParse.H>

using namespace amrex;

/**
 * Generates settings object from input file and verbosity of AmrLevelAdv.
 */
IMEXSettings get_imex_settings(bool verbose)
{
    IMEXSettings settings;
    settings.verbose = verbose;
    ParmParse pp("imex");
    pp.query("solver_verbose", settings.solver_verbose);
    pp.query("bottom_solver_verbose", settings.bottom_solver_verbose);

    std::string adv_flux;
    pp.query("advection_flux", adv_flux);
    if (adv_flux.empty() || adv_flux == "rusanov")
        settings.advection_flux = IMEXSettings::rusanov;
    else if (adv_flux == "muscl_rusanov")
        settings.advection_flux = IMEXSettings::muscl_rusanov;
    else
    {
        amrex::Abort("Invalid advection flux specified! Valid options are "
                     "rusanov and muscl_rusanov.");
    }

    pp.get("linearise", settings.linearise);
    if (!settings.linearise)
    {
        pp.get("picard_max_iter", settings.picard_max_iter);
        pp.get("compute_residual", settings.compute_residual);
        pp.query("res_tol", settings.res_tol);
    }
    else
    {
        std::string enth_method;
        pp.get("enthalpy_method", enth_method);
        if (enth_method == "reuse_pressure")
            settings.enthalpy_method = IMEXSettings::reuse_pressure;
        else if (enth_method == "conservative")
            settings.enthalpy_method = IMEXSettings::conservative;
        else if (enth_method == "conservative_ex")
            settings.enthalpy_method = IMEXSettings::conservative_ex;
        else
        {
            amrex::Abort(
                "Invalid enthalpy method specified! Valid options are "
                "reuse_pressure, conservative, and conservative_ex");
        }

        std::string ke_method;
        pp.get("ke_method", ke_method);
        if (ke_method == "conservative")
            settings.ke_method = IMEXSettings::conservative_kinetic;
        else if (ke_method == "ex")
            settings.ke_method = IMEXSettings::ex;
        else if (ke_method == "split")
            settings.ke_method = IMEXSettings::split;
        else
        {
            amrex::Abort(
                "Invalid KE method specified! Valid options are ex and split");
        }
    }

    std::string butcher_tableau;
    pp.get("butcher_tableau", butcher_tableau);
    settings.butcher_tableau
        = IMEXButcherTableau::enum_from_string(butcher_tableau);

    pp.query("stabilize", settings.stabilize);

    return settings;
}