#ifndef PlasmaSystem_H_
#define PlasmaSystem_H_

#include "System.H"
#include "System/PlasmaIndex.H"

class PlasmaSystem : public System
{
  public:
    AMREX_GPU_HOST
    virtual void ReadParameters(Parm &parm, amrex::ParmParse &pp,
                                amrex::ParmParse &pp_limiting) override;

    AMREX_GPU_HOST AMREX_FORCE_INLINE virtual int
    StateSize() const override
    {
        return PUInd::size;
    }
    AMREX_GPU_HOST AMREX_FORCE_INLINE virtual int
    ExtStateSize() const override
    {
        return PQInd::vsize;
    }

    AMREX_GPU_HOST AMREX_FORCE_INLINE virtual unsigned int
    VariableIndex(const std::string &var_name) const override;
    AMREX_GPU_HOST AMREX_FORCE_INLINE virtual bool
    VariableIsPrimitive(const std::string &var_name) const override;
    AMREX_GPU_HOST virtual amrex::Vector<std::string>
    VariableNames() const override;

    AMREX_GPU_HOST
    virtual void
    SetupDeriveList(amrex::DeriveList           &derive_lst,
                    const amrex::DescriptorList &desc_lst) const override;

    AMREX_GPU_HOST
    virtual void FillFluxes(int dir, amrex::Real time, const amrex::Box &bx,
                            const amrex::Array4<amrex::Real>       &flux,
                            const amrex::Array4<const amrex::Real> &U,
                            const Parm                             *h_parm,
                            const Parm *d_parm) const override;

    //! Templated direction for performance
    template <int dir>
    AMREX_GPU_HOST void FillFluxes(amrex::Real time, const amrex::Box &bx,
                                   const amrex::Array4<amrex::Real> &flux,
                                   const amrex::Array4<const amrex::Real> &U,
                                   const Parm *h_parm,
                                   const Parm *d_parm) const;

    /**
     * \brief Fills Q with the primitive variables calculated using U, where
     * both U and Q have StateSize() components.
     */
    AMREX_GPU_HOST
    virtual void UtoQ(const amrex::Box                       &bx,
                      const amrex::Array4<amrex::Real>       &Q,
                      const amrex::Array4<const amrex::Real> &U,
                      const Parm *h_parm, const Parm *d_parm) const override;

    /**
     * \brief Fills Q with the primitive variables calculated using U, where U
     * has StateSize() components and Q has ExtStateSize() components
     *
     * It is the responsibility of the derived class to define what the
     * additional components correspond to
     */
    AMREX_GPU_HOST
    virtual void UtoExtQ(const amrex::Box                       &bx,
                         const amrex::Array4<amrex::Real>       &ExtQ,
                         const amrex::Array4<const amrex::Real> &U,
                         const Parm                             *h_parm,
                         const Parm *d_parm) const override;

    AMREX_GPU_HOST
    virtual amrex::Real
    ExplicitTimeStepDimSplit(const amrex::Real      time,
                             const amrex::Geometry &geom,
                             const amrex::MultiFab &U, const Parm *h_parm,
                             const Parm *d_parm) const override;

    AMREX_GPU_HOST
    virtual void
    FillRBGhostCells(const amrex::Geometry &geom, amrex::MultiFab &U,
                     const amrex::MultiFab           &LS,
                     const amrex::iMultiFab          &gfm_flags,
                     RigidBodyBCType::rigidbodybctype rb_bc) const override;

  protected:
    void set_parm_defaults(Parm &parm);
    void init_parm(Parm &parm);
};

//
// Implementations
//
AMREX_GPU_HOST AMREX_FORCE_INLINE unsigned int
PlasmaSystem::VariableIndex(const std::string &var_name) const
{
    if (var_name == "density")
        return PUInd::RHO;
    else if (var_name == "mom_x")
        return PUInd::MX;
    else if (var_name == "mom_y")
        return PUInd::MY;
    else if (var_name == "mom_z")
        return PUInd::MZ;
    else if (var_name == "energy")
        return PUInd::E;
    else if (var_name == "B_x")
        return PUInd::BX;
    else if (var_name == "B_y")
        return PUInd::BY;
    else if (var_name == "B_z")
        return PUInd::BZ;
    else if (var_name == "u_x")
        return PQInd::U;
    else if (var_name == "u_y")
        return PQInd::V;
    else if (var_name == "u_z")
        return PQInd::W;
    else if (var_name == "p")
        return PQInd::P;
    else if (var_name == "T")
        return PQInd::T;
    else if (var_name == "eta")
        return PQInd::ETA;
    else if (var_name == "zeta")
        return PQInd::ZETA;
    else if (var_name == "kappa")
        return PQInd::KAPPA;
    throw amrex::RuntimeError("Invalid variable name " + var_name);
    return 0;
}
AMREX_GPU_HOST AMREX_FORCE_INLINE bool
PlasmaSystem::VariableIsPrimitive(const std::string &var_name) const
{
    if (var_name == "density" || var_name == "mom_x" || var_name == "mom_y"
        || var_name == "mom_z" || var_name == "energy" || var_name == "B_x"
        || var_name == "B_y" || var_name == "B_z")
        return false;
    else if (var_name == "u_x" || var_name == "u_y" || var_name == "u_z"
             || var_name == "p" || var_name == "T" || var_name == "eta"
             || var_name == "zeta" || var_name == "kappa")
        return true;
    else
    {
        amrex::Abort("Invalid variable name " + var_name);
        return true;
    }
}

#endif