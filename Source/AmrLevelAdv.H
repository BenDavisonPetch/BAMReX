#ifndef AmrLevelAdv_H_
#define AmrLevelAdv_H_

#include <AMReX_AmrLevel.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_FluxRegister.H>
#include <AMReX_PhysBCFunct.H>

#include "BoundaryConditions/BCs.H"
#include "IMEX/IMEXSettings.H"
#include "NumericalMethods.H"
#include "RCM/PseudoRandom.H"

#ifdef AMREX_PARTICLES
#include <AMReX_AmrParticles.H>
#endif

#include <iostream>
#include <memory>

#ifdef AMREX_USE_OMP
#include <omp.h>
#endif

// Problem-dependent struct. Defined under Exec directory.
#include "Parm.H"

enum StateType
{
    Consv_Type    = 0,
    Pressure_Type = 1,
    Levelset_Type = 2,
    NUM_STATE_TYPE
};

/**
 * AmrLevel-derived class for hyperbolic conservation equations
 */
class AmrLevelAdv : public amrex::AmrLevel
{
  public:
    /**
     * Default constructor.  Builds invalid object.
     */
    AmrLevelAdv();

    /**
     * The basic constructor.
     */
    AmrLevelAdv(amrex::Amr &papa, int lev, const amrex::Geometry &level_geom,
                const amrex::BoxArray            &bl,
                const amrex::DistributionMapping &dm, amrex::Real time);

    /**
     * The destructor.
     */
    ~AmrLevelAdv() override;

    AmrLevelAdv(AmrLevelAdv const &)                = delete;
    AmrLevelAdv &operator=(AmrLevelAdv const &)     = delete;
    AmrLevelAdv(AmrLevelAdv &&) noexcept            = delete;
    AmrLevelAdv &operator=(AmrLevelAdv &&) noexcept = delete;

    /**
     * Restart from a checkpoint file.
     */
    void restart(amrex::Amr &papa, std::istream &is,
                 bool bReadSpecial = false) override;

    /**
     * Write a checkpoint file.
     */
    void checkPoint(const std::string &dir, std::ostream &os,
                    amrex::VisMF::How how      = amrex::VisMF::NFiles,
                    bool              dump_old = true) override;

    /**
     * Write a plotfile to specified directory.
     */
    void writePlotFile(const std::string &dir, std::ostream &os,
                       amrex::VisMF::How how) override;

    /**
     * Define data descriptors.
     */
    static void variableSetUp();

    /**
     * Cleanup data descriptors at end of run.
     */
    static void variableCleanUp();

    /**
     * Initialize grid data at problem start-up.
     */
    void initData() override;

    /**
     * Initialize data on this level from another AmrLevelAdv (during regrid).
     */
    void init(amrex::AmrLevel &old) override;

    /**
     * Initialize data on this level after regridding if old level did not
     * previously exist
     */
    void init() override;

    /**
     * Advance grids at this level in time.
     */
    amrex::Real advance(amrex::Real time, amrex::Real dt, int iteration,
                        int ncycle) override;

    /**
     * Advance when using a fully explicit method (e.g. MHM, HLLC)
     */
    void
    advance_explicit(amrex::Real time, amrex::Real dt,
                     amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> &fluxes,
                     amrex::MultiFab &Sborder1, amrex::MultiFab &Sborder2,
                     amrex::MultiFab &S_new);

    /**
     * Estimate time step.
     */
    amrex::Real estTimeStep(amrex::Real dt_old);

    /**
     * Compute initial time step.
     */
    amrex::Real initialTimeStep();

    /**
     * Compute initial `dt'.
     */
    void computeInitialDt(int finest_level, int sub_cycle,
                          amrex::Vector<int>                  &n_cycle,
                          const amrex::Vector<amrex::IntVect> &ref_ratio,
                          amrex::Vector<amrex::Real>          &dt_level,
                          amrex::Real stop_time) override;

    /**
     * Compute new `dt'.
     */
    void computeNewDt(int finest_level, int sub_cycle,
                      amrex::Vector<int>                  &n_cycle,
                      const amrex::Vector<amrex::IntVect> &ref_ratio,
                      amrex::Vector<amrex::Real>          &dt_min,
                      amrex::Vector<amrex::Real>          &dt_level,
                      amrex::Real stop_time, int post_regrid_flag) override;

    /**
     * Do work after timestep().
     */
    void post_timestep(int iteration) override;

    /**
     * Do work after regrid().
     */
    void post_regrid(int lbase, int new_finest) override;

    /**
     * Do work after a restart().
     */
    void post_restart() override;

    /**
     * Do work after init().
     */
    void post_init(amrex::Real stop_time) override;

    /**
     * Error estimation for regridding.
     */
    void errorEst(amrex::TagBoxArray &tags, int clearval, int tagval,
                  amrex::Real time, int n_error_buf = 0,
                  int ngrow = 0) override;

#ifdef AMREX_PARTICLES
    static amrex::AmrTracerParticleContainer *theTracerPC()
    {
        return TracerPC.get();
    }
#endif

    const static int NUM_STATE;
    const static int NUM_GROW;
    const static int NUM_GROW_P;

    // Parm class instances
    static Parm *h_parm;
    static Parm *d_parm;

#if REUSE_PRESSURE
    static amrex::MultiFab MFpressure;
#endif

    // Variables for error tagging
    static int                        max_phierr_lev;
    static int                        max_phigrad_lev;
    static int                        max_int_lev;
    static int                        max_box_lev; // refine within a box
    static amrex::Vector<amrex::Real> phierr;
    static unsigned int               phierr_var_idx;
    static bool                       phierr_var_is_primv;
    static amrex::Vector<amrex::Real> phigrad;
    static unsigned int               phigrad_var_idx;
    static bool                       phigrad_var_is_primv;
    static amrex::Vector<amrex::Real> lstol;
    static amrex::RealBox             refine_box;

#if !defined(AMREX_USE_CUDA)
  protected:
#endif

    // Function to read parameters from input file.
    static void read_params();

    /**
     * Function to read tagging parameters from input file.
     * See Tagging_params.cpp for implementation.
     */
    static void get_tagging_params();

    /**
     * Function to perform advection for a single level.
     * See Adv.cpp for implementation.
     */
    AMREX_GPU_HOST
    static void
    advect(amrex::Real /*time*/, const amrex::Box &bx,
           amrex::GpuArray<amrex::Box, BL_SPACEDIM> nbx,
           const amrex::FArrayBox &statein, amrex::FArrayBox &stateout,
           AMREX_D_DECL(const amrex::FArrayBox &vx, const amrex::FArrayBox &vy,
                        const amrex::FArrayBox &vz),
           AMREX_D_DECL(amrex::FArrayBox &fx, amrex::FArrayBox &fy,
                        amrex::FArrayBox &fz),
           amrex::GpuArray<amrex::Real, BL_SPACEDIM> const &dx,
           amrex::Real                                      dt);

    // inline functions implemented below
    AmrLevelAdv         &getLevel(int lev);
    amrex::FluxRegister &getFluxReg();
    amrex::FluxRegister &getFluxReg(int lev);

    void reflux();

    void avgDown();

    void avgDown(int state_indx);

    /**
     * Initialises level set. Implemented in AmrLevelAdv_LSinit.cpp
     */
    void init_levelset(amrex::MultiFab &LS) const;

    /*
     * The data.
     */
    std::unique_ptr<amrex::FluxRegister> flux_reg;
    PseudoRandom<>                       random; // used for RCM
    amrex::iMultiFab                     gfm_flags;

    /*
     * Static data members.
     */
    static int                      verbose;
    static amrex::Real              cfl;
    static int                      do_reflux;
    static amrex::Real              acoustic_timestep_end_time;
    static NumericalMethods::Method num_method;
    static ViscMethods::Method      visc;
    static IMEXSettings             imex_settings;
    static bamrexBCData             bc_data;
    static int rot_axis, alpha; // data for geometric source term

#ifdef AMREX_PARTICLES
    void                                                      init_particles();
    static int                                                do_tracers;
    static std::unique_ptr<amrex::AmrTracerParticleContainer> TracerPC;
#endif
};

/*
 * Inlines.
 */

inline AmrLevelAdv &AmrLevelAdv::getLevel(int lev)
{
    return *(AmrLevelAdv *)&parent->getLevel(lev);
}

inline amrex::FluxRegister &AmrLevelAdv::getFluxReg()
{
    BL_ASSERT(flux_reg);
    return *flux_reg;
}

inline amrex::FluxRegister &AmrLevelAdv::getFluxReg(int lev)
{
    return getLevel(lev).getFluxReg();
}

#endif /*_AmrLevelAdv_H_*/
