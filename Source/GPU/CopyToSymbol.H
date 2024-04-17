#ifndef CopyToSymbol_H_
#define CopyToSymbol_H_

#include <cstring>
#include <iterator>

#ifdef AMREX_USE_GPU
#ifndef AMREX_USE_CUDA
#error "Functions are unimplemented for GPU frameworks that are not CUDA"
#endif
#endif

namespace bamrex
{
namespace gpu
{

template <class InIter, class OutIter>
void copy_to_symbol(OutIter symbol, InIter start, InIter end)
{
    using value_type = typename std::iterator_traits<InIter>::value_type;
    using out_value_type = typename std::iterator_traits<OutIter>::value_type;

    static_assert(std::is_same<value_type, out_value_type>::value, "");
    static_assert(std::is_trivially_copyable<value_type>(),
                  "Can only copy trivially copyable types");

    auto size = std::distance(start, end);
    if (size == 0)
        return;

#ifdef AMREX_USE_GPU
    AMREX_CUDA_SAFE_CALL(cudaMemcpyToSymbol(*symbol, &(*start), size * sizeof(value_type)));
#else
    std::memcpy(&(*symbol), &(*start), size * sizeof(value_type));
#endif
}

};
};

#endif