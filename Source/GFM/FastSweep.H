#ifndef FastSweep_H_
#define FastSweep_H_

#include <AMReX_Array4.H>
#include <AMReX_Box.H>
#include <AMReX_Extension.H>

#include "GFMFlag.H"

/**
 * \brief Generic fast sweeping function.
 *
 * \param bx
 * \param flag GFM flags
 * \param operation should have signature void(int,int,int). Will be called on
 * all cells that are not at the interface
 */
template <class F, unsigned int dim = AMREX_SPACEDIM>
AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE void
fast_sweep(const amrex::Box &bx, const amrex::Array4<const int> &flag,
           F operation)
{
    //
    // 1D
    //
#if AMREX_SPACEDIM == 1
    (void)post_inner_loop_op;
    const int k = bx.loVect3d()[2];
    const int j = bx.loVect3d()[1];
    // i = lo : hi
    for (int i = bx.loVect3d()[0]; i <= bx.hiVect3d()[0]; ++i)
    {
        if (at_interface(flag(i, j, k)))
            continue;
        operation(i, j, k);
    }
    // post_sweep_op(1);
    // i = hi : lo
    for (int i = bx.hiVect3d()[0]; i >= bx.loVect3d()[0]; --i)
    {
        if (at_interface(flag(i, j, k)))
            continue;
        operation(i, j, k);
    }
    // post_sweep_op(2);
#elif AMREX_SPACEDIM == 2
    //
    // 2D
    //
    const int k = bx.loVect3d()[2];
    // j = lo : hi, i = lo : hi
    for (int j = bx.loVect3d()[1]; j <= bx.hiVect3d()[1]; ++j)
    {
        for (int i = bx.loVect3d()[0]; i <= bx.hiVect3d()[0]; ++i)
        {
            if (at_interface(flag(i, j, k)))
                continue;
            operation(i, j, k);
        }
        // post_inner_loop_op(1, j);
    }
    // post_sweep_op(1);
    // j = lo : hi, i = hi : lo
    for (int j = bx.loVect3d()[1]; j <= bx.hiVect3d()[1]; ++j)
    {
        for (int i = bx.hiVect3d()[0]; i >= bx.loVect3d()[0]; --i)
        {
            if (at_interface(flag(i, j, k)))
                continue;
            operation(i, j, k);
        }
        // post_inner_loop_op(2, j);
    }
    // post_sweep_op(2);
    // j = hi : lo, i = hi : lo
    for (int j = bx.hiVect3d()[1]; j >= bx.loVect3d()[1]; --j)
    {
        for (int i = bx.hiVect3d()[0]; i >= bx.loVect3d()[0]; --i)
        {
            if (at_interface(flag(i, j, k)))
                continue;
            operation(i, j, k);
        }
        // post_inner_loop_op(3, j);
    }
    // post_sweep_op(3);
    // j = hi : lo, i = lo : hi
    for (int j = bx.hiVect3d()[1]; j >= bx.loVect3d()[1]; --j)
    {
        for (int i = bx.loVect3d()[0]; i <= bx.hiVect3d()[0]; ++i)
        {
            if (at_interface(flag(i, j, k)))
                continue;
            operation(i, j, k);
        }
        // post_inner_loop_op(4, j);
    }
    // post_sweep_op(4);
#elif AMREX_SPACEDIM == 3
    amrex::Abort("Fast sweeping for spacedim 3 not supported yet!");
#endif
}

#endif