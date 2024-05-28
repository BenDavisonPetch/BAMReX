#include <AMReX_MultiFab.H>

using namespace amrex;

void add_with_saxpy(const Vector<MultiFab> &fabs,
                    const Vector<Real> &weighting, MultiFab &soln);
template <size_t size>
void add_with_loop(const Vector<MultiFab> &fabs, const Vector<Real> &weighting,
                   MultiFab &soln);

int main(int argc, char *argv[])
{
    Initialize(argc, argv);

    const int SIZE          = 4;
    const int NCOMP         = 4;
    const int NCELLS        = 200;
    const int MAX_GRID_SIZE = 50;

    IntVect  dom_lo(AMREX_D_DECL(0, 0, 0));
    IntVect  dom_hi(AMREX_D_DECL(NCELLS - 1, NCELLS - 1, NCELLS - 1));
    Box      domain(dom_lo, dom_hi);
    BoxArray ba(domain);
    ba.maxSize(MAX_GRID_SIZE);
    DistributionMapping dm(ba);

    Vector<MultiFab> fabs(SIZE);
    Vector<Real>     weighting(SIZE);
    MultiFab         soln;
    for (int i = 0; i < SIZE; ++i)
    {
        fabs[i].define(ba, dm, NCOMP, 0);
        weighting[i] = i / ((Real)SIZE * (SIZE - 1) / 2);
    }
    soln.define(ba, dm, NCOMP, 0);

    add_with_saxpy(fabs, weighting, soln);
    add_with_loop<SIZE>(fabs, weighting, soln);

    Finalize();
}

void add_with_saxpy(const Vector<MultiFab> &fabs,
                    const Vector<Real> &weighting, MultiFab &soln)
{
    BL_PROFILE("add_with_saxpy()");
    soln.setVal(0);
    for (int n = 0; n < fabs.size(); ++n)
    {
        amrex::Saxpy(soln, weighting[n], fabs[n], 0, 0, soln.nComp(),
                     soln.n_grow);
    }
}

template <size_t size>
void add_with_loop(const Vector<MultiFab> &fabs, const Vector<Real> &weighting,
                   MultiFab &soln)
{
    BL_PROFILE("add_with_loop()");

    AMREX_ASSERT(fabs.size() == size);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    {
        GpuArray<Array4<const Real>, size> srcarr;
        GpuArray<Real, size>               weightarr;
        for (size_t i = 0; i < size; ++i) weightarr[i] = weighting[i];

        for (MFIter mfi(soln, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const auto &bx  = mfi.growntilebox();
            const auto &dst = soln.array(mfi);
            for (size_t i = 0; i < size; ++i)
            {
                srcarr[i] = fabs[i].const_array(mfi);
            }

            ParallelFor(bx, soln.nComp(),
                        [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                        {
                            dst(i, j, k, n)
                                = srcarr[0](i, j, k, n) * weightarr[0];
                            for (size_t iN = 1; iN < size; ++iN)
                            {
                                dst(i, j, k, n)
                                    += srcarr[iN](i, j, k, n) * weightarr[iN];
                            }
                        });
        }
    }
}