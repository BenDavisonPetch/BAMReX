#ifndef BCs_H_
#define BCs_H_

#include "Euler/NComp.H"
#include <AMReX.H>
#include <AMReX_Array.H>
#include <AMReX_BCRec.H>
#include <AMReX_LO_BCTYPES.H>
#include <AMReX_ParmParse.H>
#include <AMReX_StateDescriptor.H>

#include <string>

class bamrexBCData
{
  public:
    /**
     * BC type definitions
     */
    enum BCType
    {
        transmissive,
        dirichlet,
        reflective,
        periodic,
        gresho
    };

    static constexpr int CONSV_NCOMP = EULER_NCOMP;

    /**
     * Helper functions
     */
    static inline BCType enum_from_string(const std::string &name)
    {
        if (name == "transmissive")
            return transmissive;
        else if (name == "dirichlet")
            return dirichlet;
        else if (name == "reflective")
            return reflective;
        else if (name == "periodic")
            return periodic;
        else if (name == "gresho")
            return gresho;
        else
        {
            amrex::Abort("Invalid BC type specified");
            return transmissive;
        }
    }

    /**
     * Default constructor. Builds an invalid object
     */
    bamrexBCData();

    /**
     * Used to initialise the object
     */
    void build(const amrex::Geometry *top_geom);

    /**
     * Used to fill boundaries on pressure
     */
    void fill_pressure_boundary(const amrex::Geometry &geom,
                                amrex::MultiFab       &pressure,
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
    inline const amrex::BCRec &get_pressure_bcrec() const
    {
        return pressure_bcrec;
    }
    inline const amrex::BCRec &get_consv_bcrec(int comp) const
    {
        AMREX_ASSERT(comp >= 0 && comp < CONSV_NCOMP);
        return consv_bcrecs[comp];
    }
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
    //! Whether or not there are dirichlet conditions anywhere on the domain
    //! boundary
    inline bool domain_has_dirichlet() const
    {
        return m_dirichlet || m_gresho;
    }

  protected:
    amrex::Array<BCType, 2 * AMREX_SPACEDIM> bcs;

    // Boundary functions
    amrex::BndryFuncFabDefault        bndryfunc_consv, bndryfunc_p;
    amrex::StateDescriptor::BndryFunc bndryfunc_consv_sd, bndryfunc_p_sd;

    // BCRecs
    amrex::Vector<amrex::BCRec> consv_bcrecs;
    amrex::BCRec                pressure_bcrec;

    // LinOp BCs (used for IMEX)
    amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM> p_linop_lobc,
        p_linop_hibc;

    bool m_dirichlet, m_gresho;
};

#endif