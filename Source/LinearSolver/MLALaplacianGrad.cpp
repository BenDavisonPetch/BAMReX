#include "MLALaplacianGrad.H"
#include "MLALaplacianGradAlt.H"

namespace amrex
{

template class MLABecLaplacianGradT<MultiFab>;
template class MLABecLaplacianGradAltT<MultiFab>;

}