/**
 * Provides classes for boundary filling where the MultiFab to be filled has
 * BCs dependent on the information in another MultiFab. Used for IMEX pressure
 * BCs, for example.
 *
 * MultiBCFunct is equivalent to amrex's PhysBCFunct, and can be cast to
 * MultiBCFunct to be passed around as a simple function object
 * MultiCCBndryFuncFab is equivalent to amrex's GpuBndryFuncFab and uses
 * amrex's routines for filling foextrap, hoextrap, hoextrapcc, reflect_even,
 * and reflect_odd boundaries. It is specialised for cell centred MultiFabs.
 *
 * Much of the code is replicated from AMReX_PhysBCFunct.H
 *
 * For usage example see bc_jet.cpp
 */
#ifndef MultiBCFunct_H_
#define MultiBCFunct_H_

#include <AMReX_BCRec.H>
#include <AMReX_FilCC_C.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

namespace bamrex
{

using namespace amrex;

using MultiCCBndryFuncFabDefault = std::function<void(Box const& bx, FArrayBox& data, const FArrayBox& extra,
                                                   int dcomp, int numcomp,
                                                   Geometry const& geom, Real time,
                                                   const Vector<BCRec>& bcr, int bcomp,
                                                   int scomp)>;

/**
 * F must have an operator() with signature (const Box&, FArrayBox& dest, const
 * FArrayBox& extra, int icomp, int ncomp, const Geometry& geom, Real time,
 * const Vector<BCRec>& bcrs, int bccomp)
 */
template <class F> class MultiBCFunct
{
  public:
    MultiBCFunct() = default;

    MultiBCFunct(const Geometry &geom, const Vector<BCRec> &bcr, F const &f)
        : m_geom(geom)
        , m_bcr(bcr)
        , m_f(f)
    {
    }

    MultiBCFunct(const Geometry &geom, const Vector<BCRec> &bcr, F &&f)
        : m_geom(geom)
        , m_bcr(bcr)
        , m_f(std::move(f))
    {
    }

    void define(const Geometry &geom, const Vector<BCRec> &bcr, F const &f)
    {
        m_geom = geom;
        m_bcr  = bcr;
        m_f    = f;
    }

    void define(const Geometry &geom, const Vector<BCRec> &bcr, F &&f)
    {
        m_geom = geom;
        m_bcr  = bcr;
        m_f    = std::move(f);
    }

    void operator()(MultiFab &mf, const MultiFab &extra, int icomp, int ncomp,
                    IntVect const &nghost, Real time, int bccomp)
    {
        if (m_geom.isAllPeriodic())
        {
            return;
        }

        BL_PROFILE("MultiBCFunct::()");

        //! create a grown domain box containing valid + periodic cells
        const Box &domain  = m_geom.Domain();
        Box        gdomain = amrex::convert(domain, mf.boxArray().ixType());
        for (int i = 0; i < AMREX_SPACEDIM; ++i)
        {
            if (m_geom.isPeriodic(i))
            {
                gdomain.grow(i, nghost[i]);
            }
        }

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        {
            Vector<BCRec> bcrs(ncomp);
            for (MFIter mfi(mf); mfi.isValid(); ++mfi)
            {
                FArrayBox       &dest     = mf[mfi];
                const FArrayBox &extrafab = extra[mfi];
                const Box       &bx       = mfi.fabbox();

                //! if there are cells not in the valid + periodic grown box
                //! we need to fill them here
                //!
                if (!gdomain.contains(bx))
                {
                    //! Based on BCRec for the domain, we need to make BCRec
                    //! for this Box
                    amrex::setBC(bx, domain, bccomp, 0, ncomp, m_bcr, bcrs);

                    //! Note that we pass 0 as starting component of bcrs.
                    m_f(bx, dest, extrafab, icomp, ncomp, m_geom, time, bcrs,
                        0, bccomp);
                }
            }
        }
    }

  private:
    Geometry      m_geom;
    Vector<BCRec> m_bcr;
    F             m_f;
};

/**
 * F must have an operator() with signature (const IntVect&, const
 * Array4<Real>& dst, const Array4<const Real>& extra, int dcomp, int numcomp,
 * GeometryData geom_data, Real time, const BCRec* bcr, int bcomp, int
 * orig_comp) bccomp is passed from MultiCCBndryFuncFab, which is set by the
 * bccomp argument in MultiBCFunct::(). bcomp is always 0
 */
template <class F> class MultiCCBndryFuncFab
{
  public:
    MultiCCBndryFuncFab() = default;
    MultiCCBndryFuncFab(F const &a_f)
        : m_user_f(a_f)
    {
    }
    MultiCCBndryFuncFab(F &&a_f)
        : m_user_f(std::move(a_f))
    {
    }

    void operator()(Box const &bx, FArrayBox &dest, const FArrayBox &extra,
                    int dcomp, int numcomp, Geometry const &geom, Real time,
                    const Vector<BCRec> &bcr, int bcomp, int orig_comp);

    template <class FF>
    void ccfcdoit(Box const &bx, FArrayBox &dest, const FArrayBox &extra,
                  int dcomp, int numcomp, Geometry const &geom, Real time,
                  const Vector<BCRec> &bcr, int bcomp, int orig_comp,
                  FF const &fillfunc);

  protected:
    F m_user_f;
};

template <class F>
void MultiCCBndryFuncFab<F>::operator()(Box const &bx, FArrayBox &dest,
                                        const FArrayBox &extra, int dcomp,
                                        int numcomp, Geometry const &geom,
                                        Real time, const Vector<BCRec> &bcr,
                                        int bcomp, int orig_comp)
{
#if defined(__CUDACC__) && (__CUDACC_VER_MAJOR__ == 11)                       \
    && (__CUDACC_VER_MINOR__ == 6)
    // The compiler is allowed to always reject static_assert(false,..) even
    // without instantiating this function.  The following is to work around
    // the issue.  Now the compiler is not allowed to reject the code unless
    // GpuBndryFuncFab::operator() is instantiated.
    static_assert(
        std::is_same<F, int>::value,
        "CUDA 11.6 bug: https://github.com/AMReX-Codes/amrex/issues/2607");
#endif
    AMREX_ASSERT(bx.ixType().cellCentered());
    ccfcdoit(bx, dest, extra, dcomp, numcomp, geom, time, bcr, bcomp,
             orig_comp, FilccCell{});
}

template <class F>
template <class FF>
void MultiCCBndryFuncFab<F>::ccfcdoit(Box const &bx, FArrayBox &dest,
                                      const FArrayBox &extra, int dcomp,
                                      int numcomp, Geometry const &geom,
                                      Real time, const Vector<BCRec> &bcr,
                                      int bcomp, int orig_comp,
                                      FF const &fillfunc)
{
    const IntVect &len = bx.length();

    IndexType  idxType = bx.ixType();
    Box const &domain  = amrex::convert(geom.Domain(), idxType);
    Box        gdomain = domain;
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
    {
        if (geom.isPeriodic(idim))
        {
            gdomain.grow(idim, len[idim]);
        }
    }

    if (gdomain.contains(bx))
    {
        return;
    }

    Array4<Real> const       &fab      = dest.array();
    Array4<const Real> const &extrafab = extra.const_array();
    const auto                geomdata = geom.data();

#ifdef AMREX_USE_GPU
    AsyncArray<BCRec> bcr_aa(bcr.data() + bcomp, numcomp);
    BCRec            *bcr_p = bcr_aa.data();

    const auto f_user = m_user_f;

    // fill on the faces first
    {
        Array<Box, 2 *AMREX_SPACEDIM> dom_face_boxes = {
            AMREX_D_DECL(
                amrex::convert(amrex::adjCellLo(gdomain, 0, len[0]), idxType),
                amrex::convert(amrex::adjCellLo(gdomain, 1, len[1]), idxType),
                amrex::convert(amrex::adjCellLo(gdomain, 2, len[2]), idxType)),
            AMREX_D_DECL(
                amrex::convert(amrex::adjCellHi(gdomain, 0, len[0]), idxType),
                amrex::convert(amrex::adjCellHi(gdomain, 1, len[1]), idxType),
                amrex::convert(amrex::adjCellHi(gdomain, 2, len[2]), idxType))
        };

        Vector<Box> face_boxes;
        for (const Box &b : dom_face_boxes)
        {
            Box tmp = b & bx;
            if (tmp.ok())
            {
                face_boxes.push_back(tmp);
            }
        }
        const int n_face_boxes = face_boxes.size();
        if (n_face_boxes == 1)
        {
            amrex::ParallelFor(
                face_boxes[0],
                [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                {
                    AMREX_D_PICK(amrex::ignore_unused(j, k),
                                 amrex::ignore_unused(k), (void)0);
                    IntVect const idx(AMREX_D_DECL(i, j, k));
                    fillfunc(idx, fab, dcomp, numcomp, domain, bcr_p, 0);
                    f_user(idx, fab, extrafab, dcomp, numcomp, geomdata, time,
                           bcr_p, 0, orig_comp);
                });
        }
        else if (n_face_boxes > 1)
        {
            AsyncArray<Box> face_boxes_aa(face_boxes.data(), n_face_boxes);
            Box            *boxes_p = face_boxes_aa.data();
            Long            ncounts = 0;
            for (const auto &b : face_boxes)
            {
                ncounts += b.numPts();
            }
            amrex::ParallelFor(
                ncounts,
                [=] AMREX_GPU_DEVICE(Long icount) noexcept
                {
                    const auto &idx = getCell(boxes_p, n_face_boxes, icount);
                    fillfunc(idx, fab, dcomp, numcomp, domain, bcr_p, 0);
                    f_user(idx, fab, extrafab, dcomp, numcomp, geomdata, time,
                           bcr_p, 0, orig_comp);
                });
        }
    }

#if (AMREX_SPACEDIM >= 2)
    // fill on the box edges
    {
#if (AMREX_SPACEDIM == 2)
        Array<Box, 4> dom_edge_boxes = {
            {amrex::convert(
amrex::adjCellLo(amrex::adjCellLo(gdomain, 0, len[0]), 1,
             len[1]),
             idxType),
             amrex::convert(
             amrex::adjCellLo(amrex::adjCellHi(gdomain, 0, len[0]), 1,
             len[1]),
             idxType),
             amrex::convert(
             amrex::adjCellHi(amrex::adjCellLo(gdomain, 0, len[0]), 1,
             len[1]),
             idxType),
             amrex::convert(
             amrex::adjCellHi(amrex::adjCellHi(gdomain, 0, len[0]), 1,
             len[1]),
             idxType)}
        };
#else
        Array<Box, 12> dom_edge_boxes = {
            {amrex::convert(
amrex::adjCellLo(amrex::adjCellLo(gdomain, 0, len[0]), 1,
             len[1]),
             idxType),
             amrex::convert(
             amrex::adjCellLo(amrex::adjCellHi(gdomain, 0, len[0]), 1,
             len[1]),
             idxType),
             amrex::convert(
             amrex::adjCellHi(amrex::adjCellLo(gdomain, 0, len[0]), 1,
             len[1]),
             idxType),
             amrex::convert(
             amrex::adjCellHi(amrex::adjCellHi(gdomain, 0, len[0]), 1,
             len[1]),
             idxType),
             //
             amrex::convert(
             amrex::adjCellLo(amrex::adjCellLo(gdomain, 0, len[0]), 2,
             len[2]),
             idxType),
             amrex::convert(
             amrex::adjCellLo(amrex::adjCellHi(gdomain, 0, len[0]), 2,
             len[2]),
             idxType),
             amrex::convert(
             amrex::adjCellHi(amrex::adjCellLo(gdomain, 0, len[0]), 2,
             len[2]),
             idxType),
             amrex::convert(
             amrex::adjCellHi(amrex::adjCellHi(gdomain, 0, len[0]), 2,
             len[2]),
             idxType),
             //
             amrex::convert(
             amrex::adjCellLo(amrex::adjCellLo(gdomain, 1, len[1]), 2,
             len[2]),
             idxType),
             amrex::convert(
             amrex::adjCellLo(amrex::adjCellHi(gdomain, 1, len[1]), 2,
             len[2]),
             idxType),
             amrex::convert(
             amrex::adjCellHi(amrex::adjCellLo(gdomain, 1, len[1]), 2,
             len[2]),
             idxType),
             amrex::convert(
             amrex::adjCellHi(amrex::adjCellHi(gdomain, 1, len[1]), 2,
             len[2]),
             idxType)}
        };
#endif
        Vector<Box> edge_boxes;
        for (const Box &b : dom_edge_boxes)
        {
            Box tmp = b & bx;
            if (tmp.ok())
            {
                edge_boxes.push_back(tmp);
            }
        }
        const int n_edge_boxes = edge_boxes.size();
        if (n_edge_boxes == 1)
        {
            amrex::ParallelFor(
                edge_boxes[0],
                [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                {
                    AMREX_D_PICK(amrex::ignore_unused(j, k),
                                 amrex::ignore_unused(k), (void)0);
                    IntVect const idx(AMREX_D_DECL(i, j, k));
                    fillfunc(idx, fab, dcomp, numcomp, domain, bcr_p, 0);
                    f_user(idx, fab, extrafab, dcomp, numcomp, geomdata, time,
                           bcr_p, 0, orig_comp);
                });
        }
        else if (n_edge_boxes > 1)
        {
            AsyncArray<Box> edge_boxes_aa(edge_boxes.data(), n_edge_boxes);
            Box            *boxes_p = edge_boxes_aa.data();
            Long            ncounts = 0;
            for (const auto &b : edge_boxes)
            {
                ncounts += b.numPts();
            }
            amrex::ParallelFor(
                ncounts,
                [=] AMREX_GPU_DEVICE(Long icount) noexcept
                {
                    const auto &idx = getCell(boxes_p, n_edge_boxes, icount);
                    fillfunc(idx, fab, dcomp, numcomp, domain, bcr_p, 0);
                    f_user(idx, fab, extrafab, dcomp, numcomp, geomdata, time,
                           bcr_p, 0, orig_comp);
                });
        }
    }
#endif

#if (AMREX_SPACEDIM == 3)
    // fill on corners
    {
        Array<Box, 8> dom_corner_boxes = {
            {amrex::convert(
amrex::adjCellLo(
amrex::adjCellLo(amrex::adjCellLo(gdomain, 0, len[0]), 1,
             len[1]),
             2, len[2]),
             idxType),
             amrex::convert(
             amrex::adjCellLo(
             amrex::adjCellLo(amrex::adjCellHi(gdomain, 0, len[0]), 1,
             len[1]),
             2, len[2]),
             idxType),
             amrex::convert(
             amrex::adjCellLo(
             amrex::adjCellHi(amrex::adjCellLo(gdomain, 0, len[0]), 1,
             len[1]),
             2, len[2]),
             idxType),
             amrex::convert(
             amrex::adjCellLo(
             amrex::adjCellHi(amrex::adjCellHi(gdomain, 0, len[0]), 1,
             len[1]),
             2, len[2]),
             idxType),
             amrex::convert(
             amrex::adjCellHi(
             amrex::adjCellLo(amrex::adjCellLo(gdomain, 0, len[0]), 1,
             len[1]),
             2, len[2]),
             idxType),
             amrex::convert(
             amrex::adjCellHi(
             amrex::adjCellLo(amrex::adjCellHi(gdomain, 0, len[0]), 1,
             len[1]),
             2, len[2]),
             idxType),
             amrex::convert(
             amrex::adjCellHi(
             amrex::adjCellHi(amrex::adjCellLo(gdomain, 0, len[0]), 1,
             len[1]),
             2, len[2]),
             idxType),
             amrex::convert(
             amrex::adjCellHi(
             amrex::adjCellHi(amrex::adjCellHi(gdomain, 0, len[0]), 1,
             len[1]),
             2, len[2]),
             idxType)}
        };

        Vector<Box> corner_boxes;
        for (const Box &b : dom_corner_boxes)
        {
            Box tmp = b & bx;
            if (tmp.ok())
            {
                corner_boxes.push_back(tmp);
            }
        }
        const int n_corner_boxes = corner_boxes.size();
        if (n_corner_boxes == 1)
        {
            amrex::ParallelFor(
                corner_boxes[0],
                [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                {
                    IntVect const idx(AMREX_D_DECL(i, j, k));
                    fillfunc(idx, fab, dcomp, numcomp, domain, bcr_p, 0);
                    f_user(idx, fab, extrafab, dcomp, numcomp, geomdata, time,
                           bcr_p, 0, orig_comp);
                });
        }
        else if (n_corner_boxes > 1)
        {
            AsyncArray<Box> corner_boxes_aa(corner_boxes.data(),
                                            n_corner_boxes);
            Box            *boxes_p = corner_boxes_aa.data();
            Long            ncounts = 0;
            for (const auto &b : corner_boxes)
            {
                ncounts += b.numPts();
            }
            amrex::ParallelFor(
                ncounts,
                [=] AMREX_GPU_DEVICE(Long icount) noexcept
                {
                    const auto &idx = getCell(boxes_p, n_corner_boxes, icount);
                    fillfunc(idx, fab, dcomp, numcomp, domain, bcr_p, 0);
                    f_user(idx, fab, extrafab, dcomp, numcomp, geomdata, time,
                           bcr_p, 0, orig_comp);
                });
        }
    }
#endif

#else
    BCRec const *bcr_p = bcr.data() + bcomp;

    const auto &f_user = m_user_f;

    // fill on the box faces first
    {
        Array<Box, 2 *AMREX_SPACEDIM> dom_face_boxes = {
            {AMREX_D_DECL(amrex::convert(amrex::adjCellLo(gdomain, 0, len[0]),
             idxType),
             amrex::convert(amrex::adjCellLo(gdomain, 1, len[1]),
             idxType),
             amrex::convert(amrex::adjCellLo(gdomain, 2, len[2]),
             idxType)),
             AMREX_D_DECL(amrex::convert(amrex::adjCellHi(gdomain, 0, len[0]),
             idxType),
             amrex::convert(amrex::adjCellHi(gdomain, 1, len[1]),
             idxType),
             amrex::convert(amrex::adjCellHi(gdomain, 2, len[2]),
             idxType))}
        };

        for (const Box &b : dom_face_boxes)
        {
            Box tmp = b & bx;
            amrex::For(tmp,
                       [=](int i, int j, int k) noexcept
                       {
                           amrex::ignore_unused(j, k);
                           IntVect const idx(AMREX_D_DECL(i, j, k));
                           fillfunc(idx, fab, dcomp, numcomp, domain, bcr_p,
                                    0);
                           f_user(idx, fab, extrafab, dcomp, numcomp, geomdata,
                                  time, bcr_p, 0, orig_comp);
                       });
        }
    }

#if (AMREX_SPACEDIM >= 2)
    // fill on the box edges
    {
#if (AMREX_SPACEDIM == 2)
        Array<Box, 4> dom_edge_boxes = {
            {amrex::convert(
amrex::adjCellLo(amrex::adjCellLo(gdomain, 0, len[0]), 1,
             len[1]),
             idxType),
             amrex::convert(
             amrex::adjCellLo(amrex::adjCellHi(gdomain, 0, len[0]), 1,
             len[1]),
             idxType),
             amrex::convert(
             amrex::adjCellHi(amrex::adjCellLo(gdomain, 0, len[0]), 1,
             len[1]),
             idxType),
             amrex::convert(
             amrex::adjCellHi(amrex::adjCellHi(gdomain, 0, len[0]), 1,
             len[1]),
             idxType)}
        };
#else
        Array<Box, 12> dom_edge_boxes = {
            {amrex::convert(
amrex::adjCellLo(amrex::adjCellLo(gdomain, 0, len[0]), 1,
             len[1]),
             idxType),
             amrex::convert(
             amrex::adjCellLo(amrex::adjCellHi(gdomain, 0, len[0]), 1,
             len[1]),
             idxType),
             amrex::convert(
             amrex::adjCellHi(amrex::adjCellLo(gdomain, 0, len[0]), 1,
             len[1]),
             idxType),
             amrex::convert(
             amrex::adjCellHi(amrex::adjCellHi(gdomain, 0, len[0]), 1,
             len[1]),
             idxType),
             //
             amrex::convert(
             amrex::adjCellLo(amrex::adjCellLo(gdomain, 0, len[0]), 2,
             len[2]),
             idxType),
             amrex::convert(
             amrex::adjCellLo(amrex::adjCellHi(gdomain, 0, len[0]), 2,
             len[2]),
             idxType),
             amrex::convert(
             amrex::adjCellHi(amrex::adjCellLo(gdomain, 0, len[0]), 2,
             len[2]),
             idxType),
             amrex::convert(
             amrex::adjCellHi(amrex::adjCellHi(gdomain, 0, len[0]), 2,
             len[2]),
             idxType),
             //
             amrex::convert(
             amrex::adjCellLo(amrex::adjCellLo(gdomain, 1, len[1]), 2,
             len[2]),
             idxType),
             amrex::convert(
             amrex::adjCellLo(amrex::adjCellHi(gdomain, 1, len[1]), 2,
             len[2]),
             idxType),
             amrex::convert(
             amrex::adjCellHi(amrex::adjCellLo(gdomain, 1, len[1]), 2,
             len[2]),
             idxType),
             amrex::convert(
             amrex::adjCellHi(amrex::adjCellHi(gdomain, 1, len[1]), 2,
             len[2]),
             idxType)}
        };
#endif

        for (const Box &b : dom_edge_boxes)
        {
            Box tmp = b & bx;
            amrex::For(tmp,
                       [=](int i, int j, int k) noexcept
                       {
                           amrex::ignore_unused(j, k);
                           IntVect const idx(AMREX_D_DECL(i, j, k));
                           fillfunc(idx, fab, dcomp, numcomp, domain, bcr_p,
                                    0);
                           f_user(idx, fab, extrafab, dcomp, numcomp, geomdata,
                                  time, bcr_p, 0, orig_comp);
                       });
        }
    }
#endif

#if (AMREX_SPACEDIM == 3)
    // fill on box corners
    {
        Array<Box, 8> dom_corner_boxes = {
            {amrex::convert(
amrex::adjCellLo(
amrex::adjCellLo(amrex::adjCellLo(gdomain, 0, len[0]), 1,
             len[1]),
             2, len[2]),
             idxType),
             amrex::convert(
             amrex::adjCellLo(
             amrex::adjCellLo(amrex::adjCellHi(gdomain, 0, len[0]), 1,
             len[1]),
             2, len[2]),
             idxType),
             amrex::convert(
             amrex::adjCellLo(
             amrex::adjCellHi(amrex::adjCellLo(gdomain, 0, len[0]), 1,
             len[1]),
             2, len[2]),
             idxType),
             amrex::convert(
             amrex::adjCellLo(
             amrex::adjCellHi(amrex::adjCellHi(gdomain, 0, len[0]), 1,
             len[1]),
             2, len[2]),
             idxType),
             amrex::convert(
             amrex::adjCellHi(
             amrex::adjCellLo(amrex::adjCellLo(gdomain, 0, len[0]), 1,
             len[1]),
             2, len[2]),
             idxType),
             amrex::convert(
             amrex::adjCellHi(
             amrex::adjCellLo(amrex::adjCellHi(gdomain, 0, len[0]), 1,
             len[1]),
             2, len[2]),
             idxType),
             amrex::convert(
             amrex::adjCellHi(
             amrex::adjCellHi(amrex::adjCellLo(gdomain, 0, len[0]), 1,
             len[1]),
             2, len[2]),
             idxType),
             amrex::convert(
             amrex::adjCellHi(
             amrex::adjCellHi(amrex::adjCellHi(gdomain, 0, len[0]), 1,
             len[1]),
             2, len[2]),
             idxType)}
        };

        for (const Box &b : dom_corner_boxes)
        {
            Box tmp = b & bx;
            amrex::For(tmp,
                       [=](int i, int j, int k) noexcept
                       {
                           IntVect const idx(AMREX_D_DECL(i, j, k));
                           fillfunc(idx, fab, dcomp, numcomp, domain, bcr_p,
                                    0);
                           f_user(idx, fab, extrafab, dcomp, numcomp, geomdata,
                                  time, bcr_p, 0, orig_comp);
                       });
        }
    }
#endif

#endif
}

} // namespace bamrex

#endif