#ifndef BCs_H_
#define BCs_H_

#include "BoundaryConditions/bc_jet.H"
#include "System/System.H"

#include <AMReX.H>
#include <AMReX_Array.H>
#include <AMReX_BCRec.H>
#include <AMReX_LO_BCTYPES.H>
#include <AMReX_ParmParse.H>
#include <AMReX_StateDescriptor.H>

#include "BCTypes.H"

class bamrexBCData
{
  public:
    /**
     * Default constructor. Builds an invalid object
     */
    bamrexBCData();

    /**
     * Used to initialise the object
     */
    void build(const amrex::Geometry *top_geom, const System *system,
               amrex::ParmParse &pp);

    /**
     * Used to fill boundaries on pressure
     * \param geom
     * \param pressure
     * \param U conservative variables used by some boundary conditions to
     * calculate pressure
     * \param time
     */
    void fill_pressure_boundary(const amrex::Geometry &geom,
                                amrex::MultiFab       &pressure,
                                const amrex::MultiFab &U,
                                amrex::Real            time) const;

    /**
     * Used to fill boundaries on conservative variables
     */
    void fill_consv_boundary(const amrex::Geometry &geom, amrex::MultiFab &U,
                             amrex::Real time) const;

    inline const amrex::StateDescriptor::BndryFunc &get_consv_bndryfunc() const
    {
        return bndryfunc_consv_sd;
    }

    inline const amrex::StateDescriptor::BndryFunc &
    get_pressure_bndryfunc() const
    {
        return bndryfunc_p_sd;
    }

    inline const amrex::StateDescriptor::BndryFunc &get_ls_bndryfunc() const
    {
        return bndryfunc_ls_sd;
    }

    inline const amrex::Vector<amrex::BCRec> &get_consv_bcrecs() const
    {
        return consv_bcrecs;
    }

    inline const amrex::BCRec &get_pressure_bcrec() const
    {
        return pressure_bcrec;
    }

    inline const amrex::BCRec &get_ls_bcrec() const { return ls_bcrec; }

    inline const amrex::BCRec &get_normal_bcrec() const { return norm_bcrec; }

    inline const amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM> &
    get_p_linop_lobc() const
    {
        return p_linop_lobc;
    }

    inline const amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM> &
    get_p_linop_hibc() const
    {
        return p_linop_hibc;
    }

    inline bool jet() const { return m_jet; }

    inline RigidBodyBCType::rigidbodybctype rigidbody_bc() const
    {
        return rb_bc;
    }

    inline void set_rb_enabled(bool val) { m_rb_enabled = val; }

    inline bool rb_enabled() const { return m_rb_enabled; }

  protected:
    int        ncomp;
    SystemType system_type;

    amrex::Array<BAMReXBCType::bctype, 2 * AMREX_SPACEDIM> bcs;
    RigidBodyBCType::rigidbodybctype                       rb_bc;

    // Boundary functions
    amrex::BndryFuncFabDefault        bndryfunc_consv, bndryfunc_p;
    amrex::StateDescriptor::BndryFunc bndryfunc_consv_sd, bndryfunc_p_sd,
        bndryfunc_ls_sd;

    // Cursed object used to fill pressure boundaries when using the jet
    // testcase
    bamrex::MultiCCBndryFuncFabDefault bndryfunc_jet_p;

    // BCRecs
    amrex::Vector<amrex::BCRec> consv_bcrecs;
    amrex::BCRec                pressure_bcrec, ls_bcrec, norm_bcrec;

    // LinOp BCs (used for IMEX)
    amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM> p_linop_lobc,
        p_linop_hibc;

    bool m_jet, m_ss_jet, m_gresho;
    bool m_rb_enabled;
};

#endif