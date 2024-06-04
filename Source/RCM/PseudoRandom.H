#ifndef PseudoRandom_H_
#define PseudoRandom_H_

#include <array>
#include <cmath>

#include <iostream>

//! Implements the van der Corput sequence
template <int k1 = 5, int k2 = 3, std::size_t max_size = 32> class PseudoRandom
{
  public:
    PseudoRandom()
        : n(0)
        , m(0)
        , a({ 0 })
        , A({ 0 })
    {
        a[0] = 0;
    }

    inline double random();

  protected:
    std::size_t                        n;
    std::size_t                        m;
    std::array<unsigned int, max_size> a;
    std::array<unsigned int, max_size> A;
};

template <int k1, int k2, std::size_t max_size>
double PseudoRandom<k1, k2, max_size>::random()
{
    ++n;
    ++a[0];
    for (std::size_t l = 0; l <= m; ++l)
    {
        if (a[l] == k1)
        {
            a[l] = 0;
            ++a[l + 1];
            if (l == m)
                ++m;
        }
        A[l] = (k2 * a[l]) % k1;
    }

    double rand = 0;

    for (std::size_t i = 0; i <= m; ++i)
    {
        rand += A[i] * std::pow((double)k1, -(double)(i + 1));
    }

    return rand;
}

#endif