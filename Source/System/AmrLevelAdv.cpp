#include <AMReX.H>
#include <AMReX_BCUtil.H>
#include <AMReX_BC_TYPES.H>
#include <AMReX_Exception.H>
#include <AMReX_GpuMemory.H>
#include <AMReX_MFIter.H>
#include <AMReX_MFParallelFor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_SPACE.H>
#include <AMReX_StateDescriptor.H>
#include <AMReX_TagBox.H>
#include <AMReX_VisMF.H>
#include <stdexcept>

#include "AmrLevelAdv.H"
#include "AmrLevelAdv_derive.H"
#include "BoundaryConditions/BCs.H"
#include "GFM/GFMFlag.H"
#include "GFM/RigidBody.H"
#include "IMEX/IMEXSettings.H"
#include "IMEX/IMEX_O1.H"
#include "NumericalMethods.H"
#include "Prob.H"
#include "System/Euler/Euler.H"
#include "Visc/Visc.H"
#include "tagging_K.H"
//#include "Kernels.H"

using namespace amrex;

int  AmrLevelAdv::verbose = 0;
Real AmrLevelAdv::cfl
    = 0.9; // Default value - can be overwritten in settings file
int                      AmrLevelAdv::do_reflux                  = 1;
Real                     AmrLevelAdv::acoustic_timestep_end_time = 0;
NumericalMethods::Method AmrLevelAdv::num_method;
ViscMethods::Method      AmrLevelAdv::visc = ViscMethods::none;
IMEXSettings             AmrLevelAdv::imex_settings;
bamrexBCData             AmrLevelAdv::bc_data;

int AmrLevelAdv::rot_axis = -1;
int AmrLevelAdv::alpha    = 0;

// The second-order IMEX methods have the largest stencil.
// They need 3 ghost cells for conservative variables and 1 for pressure.
// This gets larger when using rigid bodies, since we fill 3 layers of ghost
// cells.
const int AmrLevelAdv::NUM_GROW = 6; // number of ghost cells.
const int AmrLevelAdv::NUM_GROW_P
    = 4; // IMEX with no rigid bodies needs 1 ghost cell for pressure, we use 3
         // for GFM

// Mechanism for getting code to work on GPU
Parm *AmrLevelAdv::h_parm = nullptr;
Parm *AmrLevelAdv::d_parm = nullptr;

/**
 * Default constructor.  Builds invalid object.
 */
AmrLevelAdv::AmrLevelAdv() = default;

/**
 * The basic constructor.
 */
AmrLevelAdv::AmrLevelAdv(Amr &papa, int lev, const Geometry &level_geom,
                         const BoxArray &bl, const DistributionMapping &dm,
                         Real time)
    : AmrLevel(papa, lev, level_geom, bl, dm, time)
    , NUM_STATE(system->StateSize()) // system is initialised in VariableSetUp
{
    if (level > 0 && do_reflux)
    {
        // Flux registers store fluxes at patch boundaries to ensure fluxes are
        // conservative between AMR levels Flux registers are only required if
        // AMR is actually used, and if flux fix up is being done (recommended)
        flux_reg = std::make_unique<FluxRegister>(grids, dmap, crse_ratio,
                                                  level, NUM_STATE);
    }
}

/**
 * The destructor.
 */
AmrLevelAdv::~AmrLevelAdv() = default;

/**
 * Restart from a checkpoint file.
 */
// AMReX can save simultion state such that if the code crashes, it
// can be restarted, from the last checkpoint output (i.e. hhalfway
// through the simulation) with different settings files parameters if
// necessary (e.g. you might change the final time to output at the
// point of the crash, and investigate what is happening).
void AmrLevelAdv::restart(Amr &papa, std::istream &is, bool bReadSpecial)
{
    AmrLevel::restart(papa, is, bReadSpecial);

    if (level > 0 && do_reflux)
    {
        flux_reg = std::make_unique<FluxRegister>(grids, dmap, crse_ratio,
                                                  level, NUM_STATE);
    }
}

/**
 * Write a checkpoint file.
 */
void AmrLevelAdv::checkPoint(const std::string &dir, std::ostream &os,
                             VisMF::How how, bool dump_old)
{
    AmrLevel::checkPoint(dir, os, how, dump_old);
}

/**
 * Write a plotfile to specified directory.
 */
// Default output format is handled automatically by AMReX, and is well
// parallelised
void AmrLevelAdv::writePlotFile(const std::string &dir, std::ostream &os,
                                VisMF::How how)
{
    if (bc_data.rb_enabled())
    {
        // Before we write first order extrapolate density into boundary for
        // better postprocessing
        auto    &U = get_new_data(Consv_Type);
        MultiFab Sborder(grids, dmap, NUM_STATE, NUM_GROW);
        FillPatcherFill(Sborder, 0, NUM_STATE, NUM_GROW,
                        state[Consv_Type].curTime(), Consv_Type, 0);
        auto &LS = get_new_data(Levelset_Type);
        fill_ghost_rb(geom, Sborder, LS, gfm_flags, RigidBodyBCType::foextrap);
        MultiFab::Copy(U, Sborder, 0, 0, NUM_STATE, 0);
    }
    AmrLevel::writePlotFile(dir, os, how);
}

/**
 * Define data descriptors.
 */
// This is how the variables in a simulation are defined.  In the case
// of the advection equation, a single variable, phi, is defined.
void AmrLevelAdv::variableSetUp()
{
    BL_ASSERT(desc_lst.size() == 0);

    // Initialize struct containing problem-specific variables
    h_parm = new Parm{};
    d_parm = (Parm *)The_Arena()->alloc(sizeof(Parm));

    // Initialize polymorphic class which determines solved set of equations
    ParmParse  pp("prob");
    SystemType sys_type = SystemType::fluid;
    pp.queryAdd("system", sys_type);
    system = System::NewSystem(pp);

    // A function which contains all processing of the settings file,
    // setting up initial data, choice of numerical methods and
    // boundary conditions
    read_params();

    const int storedGhostZones = 0;

    // Setting up a container for a variable, or vector of variables:
    // Consv_Type: Enumerator for this variable type
    // IndexType::TheCellType(): AMReX can support cell-centred and
    // vertex-centred variables (cell centred here) StateDescriptor::Point:
    // Data can be a point in time, or an interval over time (point here)
    // storedGhostZones: Ghost zones can be stored (e.g. for output). Generally
    // set to zero. NUM_STATE: Number of variables in the variable vector (1 in
    // the case of advection equation) cell_cons_interp: Controls interpolation
    // between levels - cons_interp is good for finite volume
    desc_lst.addDescriptor(Consv_Type, IndexType::TheCellType(),
                           StateDescriptor::Point, storedGhostZones,
                           system->StateSize(), &cell_cons_interp);
    desc_lst.addDescriptor(Pressure_Type, IndexType::TheCellType(),
                           StateDescriptor::Point, NUM_GROW_P, 1,
                           &cell_cons_interp);
    // Type of interpolator shouldn't matter because level set should always be
    // reinitialised on finest grid
    desc_lst.addDescriptor(Levelset_Type, IndexType::TheCellType(),
                           StateDescriptor::Point, NUM_GROW,
                           1 + AMREX_SPACEDIM, &cell_bilinear_interp);

    Geometry const *gg = AMReX::top()->getDefaultGeometry();

    // Custom class is very useful for putting all the boundary condition stuff
    // in one place
    ParmParse ppbc("bc");
    bc_data.build(gg, system, ppbc);

    // Set up variable-specific information; needs to be done for each
    // variable in NUM_STATE Consv_Type: Enumerator for the variable type
    // being set 0: Position of the variable in the variable vector. Single
    // variable for advection. phi: Name of the variable - appears in
    // output to identify what is being plotted bc: Boundary condition
    // object for this variable (defined above) bndryfunc: The boundary
    // condition function set above
    Vector<std::string> consv_names{ "density",
                                     AMREX_D_DECL("mom_x", "mom_y", "mom_z"),
                                     "energy" };
    desc_lst.setComponent(Consv_Type, 0, consv_names,
                          bc_data.get_consv_bcrecs(),
                          bc_data.get_consv_bndryfunc());
    desc_lst.setComponent(Pressure_Type, 0, "IMEX_pressure",
                          bc_data.get_pressure_bcrec(),
                          bc_data.get_pressure_bndryfunc());

    // Boundary conditions unimportant because initialisation should set
    // ghost cells properly
    desc_lst.setComponent(Levelset_Type, 0, "level_set",
                          bc_data.get_ls_bcrec(), bc_data.get_ls_bndryfunc());
    AMREX_D_TERM(desc_lst.setComponent(Levelset_Type, 1, "n_x",
                                       bc_data.get_normal_bcrec(),
                                       bc_data.get_ls_bndryfunc());
                 , desc_lst.setComponent(Levelset_Type, 2, "n_y",
                                         bc_data.get_normal_bcrec(),
                                         bc_data.get_ls_bndryfunc());
                 , desc_lst.setComponent(Levelset_Type, 3, "n_z",
                                         bc_data.get_normal_bcrec(),
                                         bc_data.get_ls_bndryfunc());)

    //! Derived quantities
    derive_lst.add("rank", IndexType::TheCellType(), 1, derive_rank,
                   the_same_box);
    derive_lst.addComponent("rank", desc_lst, Consv_Type, 0, 1);
}

/**
 * Cleanup data descriptors at end of run.
 */
void AmrLevelAdv::variableCleanUp()
{
    desc_lst.clear();
    derive_lst.clear();

    // Delete structs containing problem-specific parameters
    delete h_parm;
    // This relates to freeing parameters on the GPU too
    The_Arena()->free(d_parm);
}

/**
 * Initialize grid data at problem start-up.
 */
void AmrLevelAdv::initData()
{
    if (verbose)
        Print() << "Initialising data at level " << level << std::endl;
    MultiFab &S_new  = get_new_data(Consv_Type);
    MultiFab &P_new  = get_new_data(Pressure_Type);
    MultiFab &LS_new = get_new_data(Levelset_Type);

    init_levelset(LS_new);
    gfm_flags.define(LS_new.boxArray(), dmap, 1, LS_new.nGrow());

    // Default filling scheme is fine for MHM and HLLC
    GFMFillInfo fill_info;
    if (num_method == NumericalMethods::imex)
    {
        fill_info.fill_diagonals = true;
        if (imex_settings.advection_flux == IMEXSettings::muscl_rusanov)
            fill_info.stencil = 3;
    }
    build_gfm_flags(gfm_flags, LS_new, fill_info);
    initdata(S_new, geom);
    AMREX_ASSERT(!S_new.contains_nan());

    // Compute initial pressure from initial conditions. Need to grow the input
    // by 2
    MultiFab Sborder(grids, dmap, NUM_STATE, NUM_GROW);
    FillPatcherFill(Sborder, 0, NUM_STATE, NUM_GROW, 0, Consv_Type, 0);
    IMEXSettings settings; // can just pass default settings to
                           // compute_pressure, doesn't make a difference
    compute_pressure(Sborder, Sborder, P_new, geom, 0, settings);
    AMREX_ASSERT(!Sborder.contains_nan());
    AMREX_ASSERT(!P_new.contains_nan());

    if (verbose)
        Print() << "Done initializing the level " << level << " data "
                << std::endl;
}

/**
 * Initialize data on this level from another AmrLevelAdv (during regrid).
 */
// These are standard AMReX commands which are unlikely to need altering
void AmrLevelAdv::init(AmrLevel &old)
{
    auto *oldlev = (AmrLevelAdv *)&old;

    //
    // Create new grid data by fillpatching from old.
    //
    Real dt_new    = parent->dtLevel(level);
    Real cur_time  = oldlev->state[Consv_Type].curTime();
    Real prev_time = oldlev->state[Consv_Type].prevTime();
    Real dt_old    = cur_time - prev_time;
    setTimeLevel(cur_time, dt_old, dt_new);

    MultiFab &S_new = get_new_data(Consv_Type);
    MultiFab &P_new = get_new_data(Pressure_Type);

    const int zeroGhosts = 0;
    // FillPatch takes the data from the first argument (which contains
    // all patches at a refinement level) and fills (copies) the
    // appropriate data onto the patch specified by the second argument:
    // old: Source data
    // S_new: destination data
    // zeroGhosts: If this is non-zero, ghost zones could be filled too - not
    // needed for init routines cur_time: AMReX can attempt interpolation if a
    // different time is specified - not recommended for advection eq.
    // Consv_Type: Specify the type of data being set
    // 0: This is the first data index that is to be copied
    // NUM_STATE: This is the number of states to be copied
    FillPatch(old, S_new, zeroGhosts, cur_time, Consv_Type, 0, NUM_STATE);
    FillPatch(old, P_new, 2, cur_time, Pressure_Type, 0, 1);

    // Note: In this example above, the all states in Consv_Type (which is
    // only 1 to start with) are being copied.  However, the FillPatch
    // command could be used to create a velocity vector from a
    // primitive variable vector.  In this case, the `0' argument is
    // replaced with the position of the first velocity component in the
    // primitive variable vector, and the NUM_STATE arguement with the
    // dimensionality - this argument is the number of variables that
    // are being filled/copied, and NOT the position of the final
    // component in e.g. the primitive variable vector.

    // Level set initialisation
    MultiFab &LS_new = get_new_data(Levelset_Type);
    init_levelset(LS_new);
    gfm_flags.define(LS_new.boxArray(), dmap, 1, LS_new.nGrow());
    // Default filling scheme is fine for MHM and HLLC
    GFMFillInfo fill_info;
    if (num_method == NumericalMethods::imex)
    {
        fill_info.fill_diagonals = true;
        if (imex_settings.advection_flux == IMEXSettings::muscl_rusanov)
            fill_info.stencil = 3;
    }
    build_gfm_flags(gfm_flags, LS_new, fill_info);
}

/**
 * Initialize data on this level after regridding if old level did not
 * previously exist
 */
// These are standard AMReX commands which are unlikely to need altering

void AmrLevelAdv::init()
{
    Real dt        = parent->dtLevel(level);
    Real cur_time  = getLevel(level - 1).state[Consv_Type].curTime();
    Real prev_time = getLevel(level - 1).state[Consv_Type].prevTime();

    Real dt_old
        = (cur_time - prev_time) / (Real)parent->MaxRefRatio(level - 1);

    setTimeLevel(cur_time, dt_old, dt);
    MultiFab &S_new = get_new_data(Consv_Type);
    MultiFab &P_new = get_new_data(Pressure_Type);

    // See first init function for documentation
    // Only difference is that because the patch didn't previously exist, there
    // is no 'old' entry (instead the function will interpolate from the coarse
    // level)
    FillCoarsePatch(S_new, 0, cur_time, Consv_Type, 0, NUM_STATE);
    FillCoarsePatch(P_new, 0, cur_time, Pressure_Type, 0, 1, 2);

    // Level set initialisation
    MultiFab &LS_new = get_new_data(Levelset_Type);
    init_levelset(LS_new);
    gfm_flags.define(LS_new.boxArray(), dmap, 1, LS_new.nGrow());
    // Default filling scheme is fine for MHM and HLLC
    GFMFillInfo fill_info;
    if (num_method == NumericalMethods::imex)
    {
        fill_info.fill_diagonals = true;
        if (imex_settings.advection_flux == IMEXSettings::muscl_rusanov)
            fill_info.stencil = 3;
    }
    build_gfm_flags(gfm_flags, LS_new, fill_info);
}

/**
 * Estimate time step.
 */
// This function is called by all of the other time step functions in AMReX,
// and is the only one that should need modifying
//
Real AmrLevelAdv::estTimeStep(Real)
{
    BL_PROFILE("AmrLevelAdv::estTimeStep()")
    GpuArray<Real, BL_SPACEDIM> dx    = geom.CellSizeArray();
    GpuArray<Real, BL_SPACEDIM> dxinv = geom.InvCellSizeArray();

    // GpuArray<Real, BL_SPACEDIM> prob_lo  = geom.ProbLoArray();
    const Real      cur_time = state[Consv_Type].curTime();
    const MultiFab &S_new    = get_new_data(Consv_Type);

    // This is just a dummy value to start with
    Real dt_est = 1.0e+20;
    if (num_method == NumericalMethods::muscl_hancock
        || num_method == NumericalMethods::hllc
        || num_method == NumericalMethods::rcm)
    {
        const Real wave_speed
            = system->MaxWaveSpeed(cur_time, S_new, h_parm, d_parm);

        if (verbose)
            Print() << "Maximum wave speed: " << wave_speed << std::endl;

        for (unsigned int d = 0; d < amrex::SpaceDim; ++d)
        {
            dt_est = std::min(dt_est, dx[d] / wave_speed);
        }
    }
    else if (num_method == NumericalMethods::imex)
    {
        if (system->GetSystemType() != SystemType::fluid)
        {
            throw RuntimeError(
                "Cannot use IMEX on systems other than fluids!");
        }
        Real wave_speed;
        if (cur_time < acoustic_timestep_end_time)
        {
            if (verbose)
                Print() << "\tUsing acoustic time step" << std::endl;
            wave_speed = system->MaxWaveSpeed(cur_time, S_new, h_parm, d_parm);
            for (unsigned int d = 0; d < amrex::SpaceDim; ++d)
            {
                dt_est = std::min(dt_est, dx[d] / wave_speed);
            }
        }
        else
        {
            if (verbose)
                Print() << "\tUsing advective time step" << std::endl;
            dt_est = 1 / max_dx_scaled_speed(S_new, dx);
        }
    }
    else
    {
        amrex::Abort("Haven't implemented a time step estimator for given "
                     "numerical method!");
    }

    if (visc == ViscMethods::op_split_explicit)
    {

        const Real visc_eigv = max_visc_eigv(S_new);
        const Real dt_visc
            = (Real)1
              / (2 * visc_eigv
                 * (AMREX_D_TERM(dxinv[0] * dxinv[0], +dxinv[1] * dxinv[1],
                                 +dxinv[2] * dxinv[2])));
        if (dt_visc < dt_est)
        {
            if (verbose)
            {
                Print() << "WARNING: The convective timestep is "
                        << dt_est / dt_visc
                        << " times larger than the viscous timestep, so the "
                           "overall time step is being limited by the viscous "
                           "time step"
                        << std::endl;
            }
            dt_est = dt_visc;
        }
    }

    dt_est *= cfl;

    if (verbose)
    {
        amrex::Print() << "AmrLevelAdv::estTimeStep at level " << level
                       << ":  dt_est = " << dt_est << std::endl;
    }

    return dt_est;
}

/**
 * Compute initial time step.
 */
Real AmrLevelAdv::initialTimeStep() { return estTimeStep(0.0); }

/**
 * Compute initial `dt'.
 */
void AmrLevelAdv::computeInitialDt(int finest_level, int /*sub_cycle*/,
                                   Vector<int> &n_cycle,
                                   const Vector<IntVect> & /*ref_ratio*/,
                                   Vector<Real> &dt_level, Real stop_time)
{
    BL_PROFILE("AmrLevelAdv::computeInitialDt()")
    //
    // Grids have been constructed, compute dt for all levels.
    //
    // AMReX's AMR Level mode assumes that the time step only needs
    // calculating on the coarsest level - all subsequent time steps are
    // reduced by the refinement factor
    if (level > 0)
    {
        return;
    }

    Real dt_0     = 1.0e+100;
    int  n_factor = 1;
    for (int i = 0; i <= finest_level; i++)
    {
        dt_level[i] = getLevel(i).initialTimeStep();
        n_factor *= n_cycle[i];
        dt_0 = std::min(dt_0, n_factor * dt_level[i]);
    }

    //
    // Limit dt's by the value of stop_time.
    //
    const Real eps      = 0.001 * dt_0;
    Real       cur_time = state[Consv_Type].curTime();
    if (stop_time >= 0.0)
    {
        if ((cur_time + dt_0) > (stop_time - eps))
        {
            dt_0 = stop_time - cur_time;
        }
    }

    n_factor = 1;
    for (int i = 0; i <= finest_level; i++)
    {
        n_factor *= n_cycle[i];
        dt_level[i] = dt_0 / n_factor;
    }
}

/**
 * Compute new `dt'.
 */
void AmrLevelAdv::computeNewDt(int          finest_level, int /*sub_cycle*/,
                               Vector<int> &n_cycle,
                               const Vector<IntVect> & /*ref_ratio*/,
                               Vector<Real> &dt_min, Vector<Real> &dt_level,
                               Real stop_time, int post_regrid_flag)
{
    BL_PROFILE("AmrLevelAdv::computeNewDt()")
    //
    // We are at the end of a coarse grid timecycle.
    // Compute the timesteps for the next iteration.
    //
    if (level > 0)
    {
        return;
    }

    // Although we only compute the time step on the finest level, we
    // need to take information from all levels into account.  The
    // sharpest features may be smeared out on coarse levels, so not
    // using finer levels could cause instability
    for (int i = 0; i <= finest_level; i++)
    {
        AmrLevelAdv &adv_level = getLevel(i);
        dt_min[i]              = adv_level.estTimeStep(dt_level[i]);
    }

    // A couple of things are implemented to ensure that time step's
    // don't suddenly grow by a lot, as this could lead to errors - for
    // sensible mesh refinement choices, these shouldn't really change
    // anything
    if (post_regrid_flag == 1)
    {
        //
        // Limit dt's by pre-regrid dt
        //
        for (int i = 0; i <= finest_level; i++)
        {
            dt_min[i] = std::min(dt_min[i], dt_level[i]);
        }
    }
    else
    {
        //
        // Limit dt's by change_max * old dt
        //
        // static Real change_max = 1.1;
        // for (int i = 0; i <= finest_level; i++)
        // {
        //     dt_min[i] = std::min(dt_min[i], change_max * dt_level[i]);
        // }
    }

    //
    // Find the minimum over all levels
    //
    Real dt_0     = 1.0e+100;
    int  n_factor = 1;
    for (int i = 0; i <= finest_level; i++)
    {
        n_factor *= n_cycle[i];
        dt_0 = std::min(dt_0, n_factor * dt_min[i]);
    }

    //
    // Limit dt's by the value of stop_time.
    //
    const Real eps      = 0.001 * dt_0;
    Real       cur_time = state[Consv_Type].curTime();
    if (stop_time >= 0.0)
    {
        if ((cur_time + dt_0) > (stop_time - eps))
        {
            dt_0 = stop_time - cur_time;
        }
    }

    n_factor = 1;
    for (int i = 0; i <= finest_level; i++)
    {
        n_factor *= n_cycle[i];
        dt_level[i] = dt_0 / n_factor;
    }
}

/**
 * Do work after timestep().
 */
// If something has to wait until all processors have done their advance
// function, the post_timestep function is the place to put it.  Refluxing and
// averaging down are two standard examples for AMR
//
void AmrLevelAdv::post_timestep(int /*iteration*/)
{
    //
    // Integration cycle on fine level grids is complete
    // do post_timestep stuff here.
    //
    int finest_level = parent->finestLevel();

    if (do_reflux && level < finest_level)
    {
        reflux();
    }

    if (level < finest_level)
    {
        avgDown();
    }

    if (level < finest_level)
    {
        // fillpatcher on level+1 needs to be reset because data on this
        // level have changed.
        getLevel(level + 1).resetFillPatcher();
    }
}

/**
 * Do work after regrid().
 */
// Nothing normally needs doing here, but if something was calculated on a
// per-patch basis, new patches might this to be calcuated immediately
//
void AmrLevelAdv::post_regrid(int lbase, int /*new_finest*/)
{
    // Prevent warnings about unused variables
    amrex::ignore_unused(lbase);
}

/**
 * Do work after a restart().
 */
void AmrLevelAdv::post_restart()
{
    MultiFab &LS_new = get_new_data(Levelset_Type);

    gfm_flags.define(LS_new.boxArray(), dmap, 1, LS_new.nGrow());
    GFMFillInfo fill_info;
    if (num_method == NumericalMethods::imex)
    {
        fill_info.fill_diagonals = true;
        if (imex_settings.advection_flux == IMEXSettings::muscl_rusanov)
            fill_info.stencil = 3;
    }
    build_gfm_flags(gfm_flags, LS_new, fill_info);
}

/**
 * Do work after init().
 */
// Once new patches have been initialised, work may need to be done to ensure
// consistency, for example, averaging down - though for linear interpolation,
// this probably won't change anything
void AmrLevelAdv::post_init(Real /*stop_time*/)
{
    if (level > 0)
    {
        return;
    }
    //
    // Average data down from finer levels
    // so that conserved data is consistent between levels.
    //
    int finest_level = parent->finestLevel();
    for (int k = finest_level - 1; k >= 0; k--)
    {
        getLevel(k).avgDown();
    }
}

/**
 * Error estimation for regridding.
 */
//  Determine which parts of the domain need refinement
void AmrLevelAdv::errorEst(TagBoxArray &tags, int /*clearval*/, int /*tagval*/,
                           Real /*time*/, int /*n_error_buf*/, int /*ngrow*/)
{
    const MultiFab &S_new = get_new_data(Consv_Type);
    const MultiFab &LS    = get_new_data(Levelset_Type);

    // Properly fill patches and ghost cells for phi gradient check
    // (i.e. make sure there is at least one ghost cell defined)
    const int oneGhost = 1;
    MultiFab  phitmp;
    if (level < max_phigrad_lev)
    {
        // Only do this if we actually compute errors based on gradient at this
        // level
        const Real cur_time = state[Consv_Type].curTime();
        phitmp.define(S_new.boxArray(), S_new.DistributionMap(), NUM_STATE, 1);
        FillPatch(*this, phitmp, oneGhost, cur_time, Consv_Type, 0, NUM_STATE);
    }
    MultiFab const &U = (level < max_phigrad_lev) ? phitmp : S_new;

    const char tagval   = TagBox::SET;
    const char clearval = TagBox::CLEAR;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    {
        FArrayBox Q;
        for (MFIter mfi(U, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box &tilebx = mfi.tilebox();
            const auto Uarr   = U.array(mfi);
            const auto ls     = LS.const_array(mfi);
            auto       tagarr = tags.array(mfi);

            if ((level < max_phierr_lev && phierr_var_is_primv)
                || (level < max_phigrad_lev && phigrad_var_is_primv))
            {
                const int  ngrow = (level < max_phigrad_lev) ? 1 : 0;
                const Box &gtbx  = grow(tilebx, ngrow);
                Q.resize(gtbx, system->ExtStateSize(), The_Async_Arena());
                system->UtoExtQ(gtbx, Q.array(), U.const_array(mfi), h_parm,
                                d_parm);
            }

            // Tag cells with high phi.
            if (level < max_phierr_lev)
            {
                const Real  phierr_lev = phierr[level];
                const auto &arr
                    = (phierr_var_is_primv) ? Q.const_array() : Uarr;
                amrex::ParallelFor(
                    tilebx,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                        state_error(i, j, k, tagarr, arr, phierr_lev,
                                    phierr_var_idx, tagval);
                    });
            }

            // Tag cells with high phi gradient.
            if (level < max_phigrad_lev)
            {
                const Real  phigrad_lev = phigrad[level];
                const auto &arr
                    = (phigrad_var_is_primv) ? Q.const_array() : Uarr;
                amrex::ParallelFor(
                    tilebx,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                        grad_error(i, j, k, tagarr, arr, phigrad_lev,
                                   phigrad_var_idx, tagval);
                    });
            }

            if (level < max_box_lev)
            {
                const auto                    &dx      = geom.CellSizeArray();
                const auto                    &prob_lo = geom.ProbLoArray();
                const auto                    &rbx_lop = refine_box.lo();
                const auto                    &rbx_hip = refine_box.hi();
                GpuArray<Real, AMREX_SPACEDIM> rbx_lo{ AMREX_D_DECL(
                    rbx_lop[0], rbx_lop[1], rbx_lop[2]) };
                GpuArray<Real, AMREX_SPACEDIM> rbx_hi{ AMREX_D_DECL(
                    rbx_hip[0], rbx_hip[1], rbx_hip[2]) };
                amrex::ParallelFor(
                    tilebx,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                        box_error(i, j, k, tagarr, dx, prob_lo, rbx_lo, rbx_hi,
                                  tagval);
                    });
            }

            // Tag cells at interfaces
            if (bc_data.rb_enabled() && level < max_int_lev)
            {
                const Real  lstol_lev = lstol[level];
                const auto &flag      = gfm_flags.const_array(mfi);
                amrex::ParallelFor(
                    tilebx,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                        interface_error(i, j, k, tagarr, ls, flag, lstol_lev,
                                        tagval);
                    });
            }

            // Untag cells in the ghost region not at the interface
            if (bc_data.rb_enabled())
            {
                const Real  lstol_lev = lstol[level];
                const auto &flag      = gfm_flags.const_array(mfi);
                amrex::ParallelFor(
                    tilebx,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                        ghost_error(i, j, k, tagarr, ls, flag, lstol_lev,
                                    clearval);
                    });
            }
        }
    }
}

/**
 * Read parameters from input file.
 */
void AmrLevelAdv::read_params()
{
    // Make sure that this is only done once
    static bool done = false;

    if (done)
    {
        return;
    }

    done = true;

    // A ParmParse object allows settings, with the correct prefix, to be read
    // in from the settings file The prefix can help identify what a settings
    // parameter is used for AMReX has some default ParmParse names, amr and
    // geometry are two commonly needed ones
    ParmParse pp("adv");

    // A ParmParse object allows settings, with the correct prefix, to be read
    // in from the settings file The prefix can help identify what a settings
    // parameter is used for AMReX has some default ParmParse names, amr and
    // geometry are two commonly needed ones
    pp.query("v", verbose);
    pp.query("cfl", cfl);
    pp.query("do_reflux", do_reflux);
    std::string method;
    pp.get("num_method", method);
    num_method = NumericalMethods::enum_from_string(method);
    std::string visc_method;
    pp.query("visc_method", visc_method);
    if (!visc_method.empty())
        visc = ViscMethods::enum_from_string(visc_method);

    if (num_method == NumericalMethods::imex)
        imex_settings = get_imex_settings(verbose);

    {
        ParmParse pp2;
        Real      stop_time;
        pp2.get("stop_time", stop_time);
        ParmParse pp3("imex");
        acoustic_timestep_end_time = 0.1 * stop_time;
        pp3.query("acoustic_timestep_end_time", acoustic_timestep_end_time);
    }
    // Parameters in Parm (limiting and material params). Will be set
    // differently depending on underlying type of system
    ParmParse ppprob("prob");
    ParmParse pplimiting("limiting");
    system->ReadParameters(*h_parm, ppprob, pplimiting);
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, h_parm, h_parm + 1, d_parm);

    // Geometric source term info
    {
        ParmParse pp2("sym");
        pp2.query("rot_axis", rot_axis);
        pp2.query("alpha", alpha); // 1 for cylindrical, 2 for spherical
    }

    // Vector variables can be read in; these require e.g.\ pp.queryarr
    // and pp.getarr, so that the ParmParse object knows to look for
    // more than one variable

    // Geometries can be Cartesian, cylindrical or spherical - some
    // functions (e.g. divergence in linear solvers) are coded with this
    // geometric dependency
    Geometry const *gg = AMReX::top()->getDefaultGeometry();

    // This tutorial code only supports Cartesian coordinates.
    if (!gg->IsCartesian())
    {
        amrex::Abort("Please set geom.coord_sys = 0");
    }

    // Read tagging parameters from tagging block in the input file.
    // See Src_nd/Tagging_params.cpp for the function implementation.
    get_tagging_params();

    // Check if rigid bodies are enabled
    ParmParse   ppls("ls");
    std::string geom_type;
    ppls.query("geom_type", geom_type);
    bc_data.set_rb_enabled(!(geom_type.empty() || geom_type == "all_regular"));
}

//
// AMReX has an inbuilt reflux command, but we still have the freedom
// to decide what goes into it (for example, which variables are
// actually refluxed).  This also gives a little flexibility as to
// where flux registers are stored.  In this example, they are stored
// on levels [1,fine] but not level 0.
//
void AmrLevelAdv::reflux()
{
    BL_ASSERT(level < parent->finestLevel());

    const auto strt = amrex::second();

    // Call the reflux command with the appropriate data.  Because there
    // are no flux registers on the coarse level, they start from the
    // first level.  But the coarse level to the (n-1)^th are the ones
    // that need refluxing, hence the `level+1'.
    getFluxReg(level + 1).Reflux(get_new_data(Consv_Type), 1.0, 0, 0,
                                 NUM_STATE, geom);

    if (verbose)
    {
        const int IOProc = ParallelDescriptor::IOProcessorNumber();
        auto      end    = amrex::second() - strt;

        ParallelDescriptor::ReduceRealMax(end, IOProc);

        amrex::Print() << "AmrLevelAdv::reflux() at level " << level
                       << " : time = " << end << std::endl;
    }
}

//
// Generic function for averaging down - in this case it just makes sure it
// doesn't happen on the finest level
//
void AmrLevelAdv::avgDown()
{
    if (level == parent->finestLevel())
    {
        return;
    }
    // Can select which variables averaging down will happen on - only one to
    // choose from in this case!
    avgDown(Consv_Type);
}

//
// Setting up the call to the AMReX-implemented average down function
//
void AmrLevelAdv::avgDown(int state_indx)
{
    // For safety, again make sure this only happens if a finer level exists
    if (level == parent->finestLevel())
    {
        return;
    }

    // You can access data at other refinement levels, use this to
    // specify your current data, and the finer data that is to be
    // averaged down
    AmrLevelAdv &fine_lev = getLevel(level + 1);
    MultiFab    &S_fine   = fine_lev.get_new_data(state_indx);
    MultiFab    &S_crse   = get_new_data(state_indx);

    // Call the AMReX average down function:
    // S_fine: Multifab with the fine data to be averaged down
    // S_crse: Multifab with the coarse data to receive the fine data where
    // necessary fine_lev.geom:  Geometric information (cell size etc.) for the
    // fine level geom: Geometric information for the coarse level (i.e. this
    // level) 0: First variable to be averaged (as not all variables need
    // averaging down S_fine.nComp(): Number of variables to average - this can
    // be computed automatically from a multifab refRatio: The refinement ratio
    // between this level and the finer level
    amrex::average_down(S_fine, S_crse, fine_lev.geom, geom, 0, S_fine.nComp(),
                        parent->refRatio(level));
}
