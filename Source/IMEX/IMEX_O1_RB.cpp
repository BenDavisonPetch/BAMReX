/**
 * Linear solver setup for rigid body calculations
 */
#include "IMEX_O1_RB.H"
#ifdef AMREX_USE_EB
#include "BCs.H"
#include "IMEX_O1.H"

#include "AmrLevelAdv.H"

using namespace amrex;

namespace IMEXEB
{

/**
 * Sets up the pressure operator when using rigid bodies
 *
 * \param geom
 * \param enthalpy
 * \param rhs
 * \param pressure_scratch filled with the dirichlet boundary conditions and
 * used to set in the linear solver if present
 * \param pressure_op
 * \param U conservative variables used by some boundary conditions to
 * determine dirichlet BC for pressure
 * \param dt
 * \param bc_data
 * \param settings
 */
AMREX_GPU_HOST
void setup_pressure_op_rb(const amrex::Geometry &geom,
                          const amrex::MultiFab &enthalpy,
                          const amrex::MultiFab &rhs,
                          amrex::MultiFab       &pressure_scratch,
                          amrex::MLEBABecLap    &pressure_op,
                          const amrex::MultiFab &U, const amrex::Real dt,
                          const bamrexBCData &bc_data)
{
    BL_PROFILE("IMEXEB::setup_pressure_op_rb()");
    AMREX_ASSERT(enthalpy.nComp() == 1);

    auto factory = dynamic_cast<EBFArrayBoxFactory const *>(
        &(pressure_scratch.Factory()));
    AMREX_ASSERT(factory);

    pressure_op.define({ geom }, { rhs.boxArray() }, { rhs.DistributionMap() },
                       LPInfo(), { factory });

    // BCs
    details::set_pressure_domain_BC(pressure_op, geom, pressure_scratch, U,
                                    bc_data);

    // default MLEBABecLap EB BC is homogenous Neumann, compatible with
    // reflective and no slip rigid bodies
    AMREX_ASSERT(bc_data.rigidbody_bc() == RigidBodyBCType::reflective
                 || bc_data.rigidbody_bc() == RigidBodyBCType::no_slip);

    // set coefficients
    const amrex::Real eps  = AmrLevelAdv::h_prob_parm->epsilon;
    const amrex::Real adia = AmrLevelAdv::h_prob_parm->adiabatic;
    pressure_op.setScalars(1, dt * dt);
    pressure_op.setACoeffs(0, eps / (adia - 1));

    // Construct face-centred enthalpy
    Array<MultiFab, AMREX_SPACEDIM> enthalpy_f;
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        BoxArray ba = rhs.boxArray();
        ba.surroundingNodes(d);
        enthalpy_f[d].define(ba, rhs.DistributionMap(), 1, 0);
    }
    amrex::average_cellcenter_to_face(GetArrOfPtrs(enthalpy_f), enthalpy,
                                      geom);

    pressure_op.setBCoeffs(0, amrex::GetArrOfConstPtrs(enthalpy_f));
}

} // namespace IMEXEB

#endif