#include <AMReX_GpuMemory.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Vector.H>

#include "AmrLevelAdv.H"
#include "Index.H"
#include "VarNames.H"

using namespace amrex;

// Parameters for mesh refinement
int          AmrLevelAdv::max_phierr_lev  = -1;
int          AmrLevelAdv::max_phigrad_lev = -1;
int          AmrLevelAdv::max_int_lev     = -1;
Vector<Real> AmrLevelAdv::phierr;
unsigned int AmrLevelAdv::phierr_var_idx      = UInd::RHO;
bool         AmrLevelAdv::phierr_var_is_primv = false;
Vector<Real> AmrLevelAdv::phigrad;
unsigned int AmrLevelAdv::phigrad_var_idx      = UInd::RHO;
bool         AmrLevelAdv::phigrad_var_is_primv = false;
Vector<Real> AmrLevelAdv::lstol;

void AmrLevelAdv::get_tagging_params()
{
    // Use ParmParse to get number of levels from input file
    amrex::ParmParse pp("tagging");
    // Different tagging criteria can be used on different levels.
    // Here, we can choose tagging based on absolute value (phierr)
    // and/or on gradient (phigrad).  The maximum level for each can be
    // set independently.
    pp.query("max_phierr_lev", max_phierr_lev);
    pp.query("max_phigrad_lev", max_phigrad_lev);
    // We can also set the maximum level for automatic refinement at solid
    // interfaces
    pp.query("max_int_lev", max_int_lev);

    // Set default values for the error thresholds, then read from input file
    if (max_phierr_lev != -1)
    {
        // Defaul values of a large number are given, so that if something
        // is missed from the settings file, it is just ignored
        phierr.resize(max_phierr_lev, 1.0e+20);
        pp.queryarr("phierr", phierr);
        std::string var_name;
        pp.query("phierr_var", var_name);
        if (!var_name.empty())
        {
            phierr_var_idx      = var_name_to_idx(var_name);
            phierr_var_is_primv = var_is_primv(var_name);
        }
    }
    if (max_phigrad_lev != -1)
    {
        phigrad.resize(max_phigrad_lev, 1.0e+20);
        pp.queryarr("phigrad", phigrad);
        std::string var_name;
        pp.query("phigrad_var", var_name);
        if (!var_name.empty())
        {
            phigrad_var_idx      = var_name_to_idx(var_name);
            phigrad_var_is_primv = var_is_primv(var_name);
        }
    }
    if (max_int_lev != -1)
    {
        // refine when |LS| < TOL
        lstol.resize(max_int_lev, 0);
        pp.queryarr("lstol", lstol);
    }
}
