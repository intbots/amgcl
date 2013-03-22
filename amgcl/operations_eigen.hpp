#ifndef AMGCL_OPERATIONS_EIGEN_HPP
#define AMGCL_OPERATIONS_EIGEN_HPP

/*
The MIT License

Copyright (c) 2012-2013 Denis Demidov <ddemidov@ksu.ru>

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
 * \file   operations_eigen.hpp
 * \author Denis Demidov <ddemidov@ksu.ru>
 * \brief  Adaptors for Eigen types.
 */

#include <amgcl/common.hpp>

#include <Eigen/Dense>
#include <Eigen/SparseCore>

#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_arithmetic.hpp>
#include <boost/type_traits/is_base_of.hpp>

namespace amgcl {

/// Returns value type for an Eigen vector.
/** Necessary for eigen types to work with amgcl::solve() functions. */
template <typename T>
struct value_type<T,
    typename boost::enable_if< boost::is_arithmetic<typename T::Scalar> >::type
    >
{
    typedef typename T::Scalar type;
};

/// Returns inner product of two Eigen vectors.
/** Necessary for eigen types to work with amgcl::solve() functions. */
template <typename T1, typename T2>
typename T1::Scalar inner_prod(const Eigen::MatrixBase<T1> &x, const Eigen::MatrixBase<T2> &y) {
    return x.dot(y);
}

/// Returns norm of an Eigen vector.
/** Necessary for eigen types to work with amgcl::solve() functions. */
template <typename T>
typename T::Scalar norm(const Eigen::MatrixBase<T> &x) {
    return x.norm();
}

/// Clears (sets elements to zero) an Eigen vector.
/** Necessary for eigen types to work with amgcl::solve() functions. */
template <typename T>
void clear(Eigen::MatrixBase<T> &x) {
    x.setZero();
}

template <class Derived>
const Eigen::Transpose<const Derived> transpose(const Eigen::DenseBase<Derived> &m) {
    return m.derived().transpose();
}

template <class Derived>
const Eigen::internal::inverse_impl<Derived> inverse(const Eigen::DenseBase<Derived> &m) {
    return m.derived().inverse();
}

template <class Derived>
typename boost::enable_if<
    boost::is_base_of< Eigen::DenseBase<Derived>, Derived >,
    const typename Eigen::DenseBase<Derived>::ConstantReturnType
    >::type
zero() {
    return Derived::Zero();
}

template <class Derived>
typename boost::enable_if<
    boost::is_base_of< Eigen::DenseBase<Derived>, Derived >,
    const typename Eigen::MatrixBase<Derived>::IdentityReturnType
    >::type
identity() {
    return Derived::Identity();
}

} // namespace amgcl

#include <amgcl/spmat.hpp>

namespace amgcl {

namespace sparse {

/// Maps Eigen sparse matrix to format supported by amgcl.
/** No data is copied here. */
template <class Derived>
matrix_map<typename Derived::Scalar, typename Derived::Index>
map(const Eigen::SparseMatrixBase<Derived> &A) {
    return matrix_map<typename Derived::Scalar, typename Derived::Index>(
            A.derived().rows(),
            A.derived().cols(),
            A.derived().outerIndexPtr(),
            A.derived().innerIndexPtr(),
            A.derived().valuePtr()
            );
}

} // namespace sparse

} // namespace amgcl


#endif
