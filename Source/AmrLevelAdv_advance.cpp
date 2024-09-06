#include "AmrLevelAdv.H"
#include "Fluxes/Fluxes.H"
#include "Fluxes/Update.H"
#include "GFM/RigidBody.H"
#include "IMEX/IMEX_RK.H"
#include "MFIterLoop.H"
#include "NumericalMethods.H"
#include "RCM/RCM.H"
#include "SourceTerms/GeometricSource.H"
#include "Visc/Visc.H"
#include <AMReX_Arena.H>
#include <AMReX_MFIter.H>

using namespace amrex;

/**
 * Advance grids at this level in time.
 */
//  This function is the one that actually calls the flux functions.
Real AmrLevelAdv::advance(Real time, Real dt, int /*iteration*/,
                          int /*ncycle*/)
{
    BL_PROFILE("AmrLevelAdv::advance()")
    if (verbose)
        Print() << "Starting advance:" << std::endl;
    // MultiFab &S_mm = get_new_data(Consv_Type);
    MultiFab &P_mm = get_new_data(Pressure_Type);
    MultiFab &LS   = get_new_data(Levelset_Type);

    // Note that some useful commands exist - the maximum and minumum
    // values on the current level can be computed directly - here the
    // max and min of variable 0 are being calculated, and output.
    // Real maxval = S_mm.max(0);
    // Real minval = S_mm.min(0);
    // amrex::Print() << "phi max = " << maxval << ", min = " << minval
    //                << std::endl;

    // This ensures that all data computed last time step is moved from
    // `new' data to `old data' -
    for (int k = 0; k < NUM_STATE_TYPE; k++)
    {
        if (k == Levelset_Type)
            continue;
        state[k].allocOldData();
        state[k].swapTimeLevels(dt);
    }

    MultiFab &S_new = get_new_data(Consv_Type);
    MultiFab &P_new = get_new_data(Pressure_Type);

    // const Real prev_time = state[Consv_Type].prevTime();
    // const Real cur_time  = state[Consv_Type].curTime();
    // const Real ctr_time  = 0.5 * (prev_time + cur_time);

    GpuArray<Real, BL_SPACEDIM> dx = geom.CellSizeArray();

    //
    // Get pointers to Flux registers, or set pointer to zero if not there.
    //
    FluxRegister *fine    = nullptr;
    FluxRegister *current = nullptr;

    int finest_level = parent->finestLevel();

    // If we are not on the finest level, fluxes may need correcting
    // from those from finer levels.  To start this process, we set the
    // flux register values to zero
    if (do_reflux && level < finest_level)
    {
        fine = &getFluxReg(level + 1);
        fine->setVal(0.0);
    }

    // If we are not on the coarsest level, the fluxes are going to be
    // used to correct those on coarser levels.  We get the appropriate
    // flux level to include our fluxes within
    if (do_reflux && level > 0)
    {
        current = &getFluxReg(level);
    }

    // Set up a dimensional multifab that will contain the fluxes
    // MultiFab fluxes[BL_SPACEDIM];
    Array<MultiFab, AMREX_SPACEDIM> fluxes;

    // Define the appropriate size for the flux MultiFab.
    // Fluxes are defined at cell faces - this is taken care of by the
    // surroundingNodes(j) command, ensuring the size of the flux
    // storage is increased by 1 cell in the direction of the flux.
    // This is only needed if refluxing is happening, otherwise fluxes
    // don't need to be stored, just used
    for (int j = 0; j < BL_SPACEDIM; j++)
    {
        BoxArray ba = S_new.boxArray();
        ba.surroundingNodes(j);
        fluxes[j].define(ba, dmap, NUM_STATE, 0);
    }

    // State with ghost cells - this is used to compute fluxes and perform the
    // update. We have two for methods with operator splitting (the output of
    // one step becomes the input of the next and needs ghost cells)
    MultiFab Sborder(grids, dmap, NUM_STATE, NUM_GROW);
    MultiFab Sborder2(grids, dmap, NUM_STATE, NUM_GROW);
    // We use FillPatcher to do fillpatch here if we can
    // This is similar to the FillPatch function documented in init(), but
    // takes care of boundary conditions too
    FillPatcherFill(Sborder, 0, NUM_STATE, NUM_GROW, time, Consv_Type, 0);

    // fill rigid body ghost states
    if (bc_data.rb_enabled())
    {
        fill_ghost_rb(geom, Sborder, LS, gfm_flags, bc_data.rigidbody_bc());
        bc_data.fill_consv_boundary(geom, Sborder, time);
    }

    if (verbose)
        Print() << "\tFilled ghost states" << std::endl;

    MultiFab::Copy(Sborder2, Sborder, 0, 0, NUM_STATE, NUM_GROW);

    BL_PROFILE_VAR("AmrLevelAdv::flux_computation", pflux_computation);

    if (num_method == NumericalMethods::muscl_hancock
        || num_method == NumericalMethods::hllc)
        advance_explicit(time, dt, fluxes, Sborder, Sborder2, S_new);
    else if (num_method == NumericalMethods::imex)
    {
        AMREX_ASSERT_WITH_MESSAGE(
            !(imex_settings.stabilize && do_reflux
              && this->parent->maxLevel() > 0),
            "Refluxing with high-Mach number stabilisation not "
            "supported!");

        // Ghost cells are filled at the start of each RK step, so they're
        // filled witihin advance_imex_rk

        MultiFab::Copy(P_new, P_mm, 0, 0, 1, NUM_GROW_P);
        bc_data.fill_pressure_boundary(geom, P_new, Sborder, time);
        if (bc_data.rb_enabled())
        {
            fill_ghost_p_rb(geom, P_new, Sborder, LS, gfm_flags,
                            bc_data.rigidbody_bc());
        }
        advance_imex_rk_stab(time, geom, Sborder, S_new, P_new, fluxes, dt,
                             bc_data, imex_settings);

        if (rot_axis != -1 && alpha != 0)
        {
            advance_geometric(geom, dt, alpha, rot_axis, S_new, Sborder);
            MultiFab::Copy(S_new, Sborder, 0, 0, NUM_STATE, 0);
        }
    }
    else if (num_method == NumericalMethods::rcm)
    {
        Real rand = random.random();
        Real dr   = geom.CellSize(0);
        AMREX_ASSERT(fabs(geom.ProbLo(0)) < 1e-12);
        for (MFIter mfi(S_new); mfi.isValid(); ++mfi)
        {
            const auto &bx = mfi.validbox();
            advance_RCM(dt, dr, rand, h_parm->adiabatic, h_parm->epsilon, bx,
                        Sborder[mfi], S_new[mfi]);
        }
        if (rot_axis != -1 && alpha != 0)
        {
            advance_geometric(geom, dt, alpha, rot_axis, S_new, Sborder);
            MultiFab::Copy(S_new, Sborder, 0, 0, NUM_STATE, 0);
        }
    }
    else
    {
        amrex::Abort("Invalid scheme!");
    }

    BL_PROFILE_VAR_STOP(pflux_computation);

    // The fluxes now need scaling for the reflux command.
    // This scaling is by the size of the boundary through which the flux
    // passes, e.g. the x-flux needs scaling by the dy, dz and dt
    if (do_reflux)
        for (int d = 0; d < AMREX_SPACEDIM; ++d)
        {
            Real scaleFactor = dt;
            for (int scaledir = 0; scaledir < amrex::SpaceDim; ++scaledir)
            {
                // Fluxes don't need scaling by dx[d]
                if (scaledir == d)
                {
                    continue;
                }
                scaleFactor *= dx[scaledir];
            }
            // The mult function automatically multiplies entries in a multifab
            // by a scalar scaleFactor: The scalar to multiply by 0: The first
            // data index in the multifab to multiply NUM_STATE:  The total
            // number of data indices that will be multiplied
            fluxes[d].mult(scaleFactor, 0, NUM_STATE);
        }

    // Refluxing at patch boundaries.  Amrex automatically does this
    // where needed, but you need to state a few things to make sure it
    // happens correctly:
    // FineAdd: If we are not on the coarsest level, the fluxes at this level
    // will form part of the correction
    //          to a coarse level
    // CrseInit:  If we are not the finest level, the fluxes at patch
    // boundaries need correcting.  Since we
    //            know that the coarse level happens first, we initialise the
    //            boundary fluxes through this function, and subsequently
    //            FineAdd will modify things ready for the correction
    // Both functions have the same arguments:
    // First: Name of the flux MultiFab (this is done dimension-by-dimension
    // Second: Direction, to ensure the correct vertices are being corrected
    // Third: Source component - the first entry of the flux MultiFab that is
    // to be copied (it is possible that
    //        some variables will not need refluxing, or will be computed
    //        elsewhere (not in this example though)
    // Fourth: Destinatinon component - the first entry of the flux register
    // that this call to FineAdd sends to Fifth: NUM_STATE - number of states
    // being added to the flux register Sixth: Multiplier - in general, the
    // least accurate (coarsest) flux is subtracted (-1) and the most
    //        accurate (finest) flux is added (+1)

    if (do_reflux)
    {
        if (current) // current is the FluxRegister for the current level if
                     // this is not the coarsest level
        {
            for (int i = 0; i < AMREX_SPACEDIM; i++)
            {
                current->FineAdd(fluxes[i], i, 0, 0, NUM_STATE, 1.);
            }
        }
        if (fine) // fine is the FluxRegister for the next level up if this is
                  // not the finest level
        {
            for (int i = 0; i < AMREX_SPACEDIM; i++)
            {
                fine->CrseInit(fluxes[i], i, 0, 0, NUM_STATE, -1.);
            }
        }
    }

    if (verbose)
        Print() << "Advance complete!" << std::endl;
    return dt;
}

void AmrLevelAdv::advance_explicit(
    amrex::Real time, amrex::Real dt,
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> &fluxes,
    amrex::MultiFab &Sborder1, amrex::MultiFab &Sborder2,
    amrex::MultiFab &S_new)
{
    const int   ncomp     = Sborder1.nComp();
    const auto &dx        = geom.CellSizeArray();
    const auto &dxinv     = geom.InvCellSizeArray();
    const bool  do_tiling = TilingIfNotGPU() && AMREX_SPACEDIM == 3;

    MultiFab *MFin  = &Sborder1;
    MultiFab *MFout = &Sborder2;

    for (int d = 0; d < amrex::SpaceDim; d++)
    {
        if (verbose)
            Print() << "\tComputing update for direction " << d << std::endl;

        const int iOffset = (d == 0 ? 1 : 0);
        const int jOffset = (d == 1 ? 1 : 0);
        const int kOffset = (d == 2 ? 1 : 0);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        {
            FArrayBox  tileflux;
            FArrayBox *flux;
            for (MFIter mfi(Sborder1, do_tiling); mfi.isValid(); ++mfi)
            {
                const auto &bx = mfi.tilebox();
                // box on which we should copy to fluxes
                const auto &nbx = mfi.nodaltilebox(d);
                // box on which we should compute fluxes for update on bx
                const auto               &tbx  = surroundingNodes(bx, d);
                const Array4<const Real> &Uin  = MFin->const_array(mfi);
                const Array4<Real>       &Uout = MFout->array(mfi);
                if (do_tiling)
                {
                    tileflux.resize(tbx, ncomp, The_Async_Arena());
                    flux = &tileflux;
                }
                else
                {
                    flux = &(fluxes[d][mfi]);
                }
                const auto &fluxarr  = flux->array();
                const auto &cfluxarr = flux->const_array();

                if (num_method == NumericalMethods::muscl_hancock)
                    compute_MUSCL_hancock_flux(d, time, bx, fluxarr, Uin, dx,
                                               dt);
                else if (num_method == NumericalMethods::hllc)
                    compute_HLLC_flux(d, time, bx, fluxarr, Uin, dx, dt);
                else
                    throw std::logic_error("Internal error");

                // Conservative update
                ParallelFor(
                    bx, ncomp,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept
                    {
                        // Conservative update formula
                        Uout(i, j, k, n)
                            = Uin(i, j, k, n)
                              - (dt / dx[d])
                                    * (cfluxarr(i + iOffset, j + jOffset,
                                                k + kOffset, n)
                                       - cfluxarr(i, j, k, n));
                    });

                // Copy tileflux to fluxes using appropriate nodal tile box
                if (do_reflux && do_tiling)
                {
                    const auto &fdst = fluxes[d].array(mfi);
                    const auto &fsrc = tileflux.const_array();
                    ParallelFor(
                        nbx, ncomp,
                        [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                            noexcept { fdst(i, j, k, n) = fsrc(i, j, k, n); });
                }
            }
        }
        // We need to compute boundary conditions again after each update
        bc_data.fill_consv_boundary(geom, *MFout, time);
        std::swap(MFin, MFout);
        if (verbose)
            Print() << "\tDirection update complete" << std::endl;
    }

    if (visc == ViscMethods::op_split_explicit)
    {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        {
            // temporary fabs
            Array<FArrayBox, AMREX_SPACEDIM> vfluxes;
            for (MFIter mfi(Sborder1, do_tiling); mfi.isValid(); ++mfi)
            {
                const Box &bx = mfi.tilebox();
                for (int idim = 0; idim < AMREX_SPACEDIM; idim++)
                {
                    vfluxes[idim].resize(surroundingNodes(bx, idim), ncomp,
                                         The_Async_Arena());
                }
                GpuArray<Box, BL_SPACEDIM> nbx;
                AMREX_D_TERM(nbx[0] = mfi.nodaltilebox(0);
                             , nbx[1] = mfi.nodaltilebox(1);
                             , nbx[2] = mfi.nodaltilebox(2));
                compute_visc_fluxes_exp(
                    bx, AMREX_D_DECL(vfluxes[0], vfluxes[1], vfluxes[2]),
                    (*MFin)[mfi], dxinv);
                // Conservative update
                conservative_update(
                    bx, (*MFin)[mfi],
                    AMREX_D_DECL(vfluxes[0], vfluxes[1], vfluxes[2]),
                    (*MFout)[mfi], dt, dxinv);
                if (do_reflux)
                {
                    Array<FArrayBox *, AMREX_SPACEDIM> fluxfabs{ AMREX_D_DECL(
                        &(fluxes[0][mfi]), &(fluxes[1][mfi]),
                        &(fluxes[2][mfi])) };
                    add_fluxes(fluxfabs, GetArrOfConstPtrs(vfluxes), nbx);
                }
            }
        }
        std::swap(MFin, MFout);
    }

    if (rot_axis != -1 && alpha != 0 && AMREX_SPACEDIM < 3)
    {
        AMREX_ALWAYS_ASSERT(
            parent->maxLevel() == 0
            || !do_reflux); // this routine doesn't update fluxes
        advance_geometric(geom, dt, alpha, rot_axis, (*MFin), (*MFout));
        std::swap(MFin, MFout);
    }
    // The updated data is now copied to the S_new multifab.  This
    // means it is now accessible through the get_new_data command, and
    // AMReX can automatically interpolate or extrapolate between
    // layers etc. S_new: Destination Sborder: Source Third entry:
    // Starting variable in the source array to be copied (the zeroth
    // variable in this case) Fourth entry: Starting variable in the
    // destination array to receive the copy (again zeroth here)
    // NUM_STATE: Total number of variables being copied Sixth entry:
    // Number of ghost cells to be included in the copy (zero in this
    // case, since only real
    //              data is needed for S_new)
    MultiFab::Copy(S_new, *MFin, 0, 0, NUM_STATE, 0);
}