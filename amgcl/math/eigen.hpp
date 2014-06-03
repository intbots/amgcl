#ifndef AMGCL_MATH_EIGEN_HPP
#define AMGCL_MATH_EIGEN_HPP

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
 * \file   amgcl/math/eigen.hpp
 * \author Denis Demidov <dennis.demidov@gmail.com>
 * \brief  Specialization of arithmetic operations for Eigen matrices.
 */

#include <boost/type_traits.hpp>
#include <Eigen/Dense>
#include <amgcl/math/interface.hpp>

namespace amgcl {
namespace math {

template <class T>
struct scalar<
    T,
    typename boost::enable_if<
        typename boost::mpl::and_<
            typename boost::mpl::bool_<(sizeof(typename T::Scalar) > 0)>::type,
            typename boost::is_base_of<Eigen::MatrixBase<T>, T>::type
        >::type
    >::type
    >
{
    typedef typename T::Scalar type;
};

template <class T>
struct zero_impl<
    T,
    typename boost::enable_if<
        typename boost::mpl::and_<
            typename boost::mpl::bool_<(sizeof(typename T::Scalar) > 0)>::type,
            typename boost::is_base_of<Eigen::MatrixBase<T>, T>::type
        >::type
    >::type
    >
{
    static T get() {
        return T::Zero();
    }
};

template <class T>
struct one_impl<
    T,
    typename boost::enable_if<
        typename boost::mpl::and_<
            typename boost::mpl::bool_<(sizeof(typename T::Scalar) > 0)>::type,
            typename boost::is_base_of<Eigen::MatrixBase<T>, T>::type
        >::type
    >::type
    >
{
    static T get() {
        return T::Identity();
    }
};

template <class T>
struct adjoint_impl<
    T,
    typename boost::enable_if<
        typename boost::mpl::and_<
            typename boost::mpl::bool_<(sizeof(typename T::Scalar) > 0)>::type,
            typename boost::is_base_of<Eigen::MatrixBase<T>, T>::type
        >::type
    >::type
    >
{
    typedef typename T::AdjointReturnType result_type;
    static result_type get(const T &v) {
        return v.adjoint();
    }
};

} // namespace math
} // namespace amgcl

#endif
