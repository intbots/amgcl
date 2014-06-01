#ifndef AMGCL_RELAXATION_GAUSS_SEIDEL_HPP
#define AMGCL_RELAXATION_GAUSS_SEIDEL_HPP

/*
The MIT License

Copyright (c) 2012-2014 Denis Demidov <dennis.demidov@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/**
 * \file   amgcl/relaxation/gauss_seidel.hpp
 * \author Denis Demidov <dennis.demidov@gmail.com>
 * \brief  Gauss-Seidel relaxation scheme.
 */

#include <amgcl/backend/interface.hpp>
#include <amgcl/relaxation/interface.hpp>

namespace amgcl {
namespace relaxation {

template <class Backend>
struct impl<gauss_seidel, Backend> {
    struct params { };

    template <class Matrix>
    impl( const Matrix&, const params &prm, const typename Backend::params&)
    {}

    template <class Matrix, class VectorRHS, class VectorX>
    static void iteration_body(
            const Matrix &A, const VectorRHS &rhs, VectorX &x, size_t i)
    {
        typedef typename backend::row_iterator<Matrix>::type row_iterator;
        typedef typename backend::value_type<Matrix>::type val_type;

        val_type temp = rhs[i];
        val_type diag = 1;
        for (row_iterator a = backend::row_begin(A, i); a; ++a) {
            if (a.col() == i)
                diag = a.value();
            else
                temp -= a.value() * x[a.col()];
        }
        x[i] = temp / diag;
    }

    template <class Matrix, class VectorRHS, class VectorX, class VectorTMP>
    void apply_pre(
            const Matrix &A, const VectorRHS &rhs, VectorX &x, VectorTMP&, const params&
            ) const
    {
        const size_t n = backend::rows(A);
        for(size_t i = 0; i < n; ++i)
            iteration_body(A, rhs, x, i);
    }

    template <class Matrix, class VectorRHS, class VectorX, class VectorTMP>
    void apply_post(
            const Matrix &A, const VectorRHS &rhs, VectorX &x, VectorTMP&, const params&
            ) const
    {
        const size_t n = backend::rows(A);
        for(size_t i = n; i-- > 0;)
            iteration_body(A, rhs, x, i);
    }
};

} // namespace relaxation
} // namespace amgcl


#endif