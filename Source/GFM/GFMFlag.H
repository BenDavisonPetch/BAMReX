#ifndef GFMFlag_H_
#define GFMFlag_H_

#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>

constexpr int GFM_GROW = 2;

struct GFMFlag
{
    static constexpr int ghost = 1 << 0;
    static constexpr int
        interface = 1 << 1; // cell at interface (either real cell next to
                            // ghost cell or vice-versa)
    static constexpr int fill
        = 1 << 2; // ghost cell, should fill using chosen GFM (i.e. ghost cell
                  // is within fill_info.stencil of an interface)
};

struct GFMFillInfo
{
    int  stencil{ 2 };
    bool fill_diagonals{ false };
};

inline bool is_ghost(int gfm_flag) noexcept
{
    return (gfm_flag & GFMFlag::ghost);
}

inline bool at_interface(int gfm_flag) noexcept
{
    return (gfm_flag & GFMFlag::interface);
}

inline bool needs_fill(int gfm_flag) noexcept
{
    return (gfm_flag & GFMFlag::fill);
}

AMREX_GPU_HOST
void build_gfm_flags(amrex::iMultiFab &flags, const amrex::MultiFab &ls,
                     const GFMFillInfo &fill_info = GFMFillInfo());

#endif