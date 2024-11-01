#ifndef FluidSystem_H_
#define FluidSystem_H_

#include "System.H"
#include "System/Index.H"

class FluidSystem : public System
{
  public:
    AMREX_GPU_HOST
    virtual void ReadParameters(Parm &parm, amrex::ParmParse &pp,
                                amrex::ParmParse &pp_limiting) override;

    AMREX_GPU_HOST AMREX_FORCE_INLINE virtual int
    StateSize() const override
    {
        return UInd::size;
    }
    AMREX_GPU_HOST AMREX_FORCE_INLINE virtual int
    ExtStateSize() const override
    {
        return QInd::vsize;
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
FluidSystem::VariableIndex(const std::string &var_name) const
{
    if (var_name == "density")
        return UInd::RHO;
    else if (var_name == "mom_x")
        return UInd::MX;
    else if (var_name == "mom_y")
        return UInd::MY;
    else if (var_name == "mom_z")
        return UInd::MZ;
    else if (var_name == "energy")
        return UInd::E;
    else if (var_name == "u_x")
        return QInd::U;
    else if (var_name == "u_y")
        return QInd::V;
    else if (var_name == "u_z")
        return QInd::W;
    else if (var_name == "p")
        return QInd::P;
    else if (var_name == "T")
        return QInd::T;
    else if (var_name == "eta")
        return QInd::ETA;
    else if (var_name == "zeta")
        return QInd::ZETA;
    else if (var_name == "kappa")
        return QInd::KAPPA;
    amrex::Abort("Invalid variable name " + var_name);
    return 0;
}
AMREX_GPU_HOST AMREX_FORCE_INLINE bool
FluidSystem::VariableIsPrimitive(const std::string &var_name) const
{
    if (var_name == "density" || var_name == "mom_x" || var_name == "mom_y"
        || var_name == "mom_z" || var_name == "energy")
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

// AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::Real
// FluidSystem::LimitingVarVal(const amrex::Array4<const amrex::Real> &U, int
// i,
//                             int j, int k, int n) const
// {
//     if (limit_on_primv[n])
//     {
//         const auto Q = UtoQ(U, i, j, k);
//         return Q[limiter_indices[n]];
//     }
//     else
//     {
//         return U(i, j, k, limiter_indices[n]);
//     }
// }

#endif