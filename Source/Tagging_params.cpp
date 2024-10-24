#include <AMReX_GpuMemory.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Vector.H>

#include "AmrLevelAdv.H"
#include "System/Index.H"

using namespace amrex;

// Parameters for mesh refinement
int          AmrLevelAdv::max_phierr_lev  = -1;
int          AmrLevelAdv::max_phigrad_lev = -1;
int          AmrLevelAdv::max_int_lev     = -1;
int          AmrLevelAdv::max_box_lev     = -1;
Vector<Real> AmrLevelAdv::phierr;
unsigned int AmrLevelAdv::phierr_var_idx      = UInd::RHO;
bool         AmrLevelAdv::phierr_var_is_primv = false;
Vector<Real> AmrLevelAdv::phigrad;
unsigned int AmrLevelAdv::phigrad_var_idx      = UInd::RHO;
bool         AmrLevelAdv::phigrad_var_is_primv = false;
Vector<Real> AmrLevelAdv::lstol;
RealBox      AmrLevelAdv::refine_box;

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
    pp.query("max_box_lev", max_box_lev);

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
            phierr_var_idx      = system->VariableIndex(var_name);
            phierr_var_is_primv = system->VariableIsPrimitive(var_name);
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
            phigrad_var_idx      = system->VariableIndex(var_name);
            phigrad_var_is_primv = system->VariableIsPrimitive(var_name);
        }
    }
    if (max_int_lev != -1)
    {
        // refine when |LS| < TOL
        lstol.resize(max_int_lev, 0);
        pp.queryarr("lstol", lstol);
    }
    if (max_box_lev != -1)
    {
        Array<Real, 2> AMREX_D_DECL(xb, yb, zb);
        pp.get("box_x", xb);
        refine_box.setLo(0, xb[0]);
        refine_box.setHi(0, xb[1]);
#if AMREX_SPACEDIM >= 2
        pp.get("box_y", yb);
        refine_box.setLo(1, yb[0]);
        refine_box.setHi(1, yb[1]);
#endif
#if AMREX_SPACEDIM == 3
        pp.get("box_z", zb);
        refine_box.setLo(2, zb[0]);
        refine_box.setHi(2, zb[1]);
#endif
    }
}
