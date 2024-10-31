#ifndef SYSTEM_H_
#define SYSTEM_H_

#include <AMReX_Array.H>
#include <AMReX_Enum.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_REAL.H>

#include "BoundaryConditions/BCTypes.H"
#include "Fluxes/Limiters.H"
#include "System/Euler/NComp.H"
#include "System/Parm.H"

/**
 * \brief Determines governing (conservative) equation.
 *
 * \c fluid indicates solution of the Euler (scaled) equations with scaling
 * parameter \f( \epsilon = \rho_0 u_0^2 / p_0 \f)
 * \c plasma indicates solution of the ideal MHD equations
 */
AMREX_ENUM(SystemType, fluid, plasma);

class System
{
  protected:
    SystemType system;

  public:
    virtual ~System() = 0;

    /**
     * Allocate space on host memory for System object, and reads parameters.
     * Returned pointer must be deleted!
     *
     * Obtains the system type (and hence actual derived class type) from the
     * parameter \c system in the given ParmParse instance.
     */
    AMREX_GPU_HOST
    static System *NewSystem(amrex::ParmParse &pp);

    AMREX_GPU_HOST
    virtual void ReadParameters(Parm &parm, amrex::ParmParse &pp,
                                amrex::ParmParse &pp_limiting)
        = 0;

    AMREX_GPU_HOST AMREX_FORCE_INLINE SystemType GetSystemType() const
    {
        return system;
    }

    //! Size of conservative state vector
    AMREX_GPU_HOST AMREX_FORCE_INLINE virtual int StateSize() const = 0;
    //! Size of extended state vector (e.g. including viscous coefficients)
    //! It is the responsibility of the derived class to define what the extra
    //! indices mean
    AMREX_GPU_HOST AMREX_FORCE_INLINE virtual int ExtStateSize() const = 0;

    AMREX_GPU_HOST AMREX_FORCE_INLINE virtual unsigned int
    VariableIndex(const std::string &var_name) const = 0;
    AMREX_GPU_HOST AMREX_FORCE_INLINE virtual bool
    VariableIsPrimitive(const std::string &var_name) const = 0;
    AMREX_GPU_HOST virtual amrex::Vector<std::string>
    VariableNames() const = 0;

    /**
     * \brief Fills \c flux with the flux function of direction \c dir
     * evaluated using \c U at each index in the given box.
     *
     * \param dir the direction of the flux function (0=x, 1=y, 2=z)
     * \param time
     * \param bx
     * \param flux
     * \param U
     * \param h_parm A pointer to the (host-memory-allocated) parameters
     * \param d_parm A pointer to the (device-memory-allocated) parameters
     */
    AMREX_GPU_HOST
    virtual void FillFluxes(int dir, amrex::Real time, const amrex::Box &bx,
                            const amrex::Array4<amrex::Real>       &flux,
                            const amrex::Array4<const amrex::Real> &U,
                            const Parm *h_parm, const Parm *d_parm) const = 0;

    /**
     * \brief Fills Q with the primitive variables calculated using U, where
     * both U and Q have StateSize() components.
     */
    AMREX_GPU_HOST
    virtual void UtoQ(const amrex::Box                       &bx,
                      const amrex::Array4<amrex::Real>       &Q,
                      const amrex::Array4<const amrex::Real> &U,
                      const Parm *h_parm, const Parm *d_parm) const = 0;

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
                         const Parm *h_parm, const Parm *d_parm) const = 0;

    //! Maximum allowable time step when using a dimensionally split
    //! fully-explicit method
    AMREX_GPU_HOST
    virtual amrex::Real ExplicitTimeStepDimSplit(const amrex::Real      time,
                                                 const amrex::Geometry &geom,
                                                 const amrex::MultiFab &U,
                                                 const Parm            *h_parm,
                                                 const Parm *d_parm) const = 0;

    AMREX_GPU_HOST
    virtual void
    FillRBGhostCells(const amrex::Geometry &geom, amrex::MultiFab &U,
                     const amrex::MultiFab           &LS,
                     const amrex::iMultiFab          &gfm_flags,
                     RigidBodyBCType::rigidbodybctype rb_bc) const = 0;
};

#endif
