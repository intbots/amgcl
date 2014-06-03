#ifndef AMGCL_MATH_INTERFACE_HPP
#define AMGCL_MATH_INTERFACE_HPP

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
 * \file   amgcl/math/interface.hpp
 * \author Denis Demidov <dennis.demidov@gmail.com>
 * \brief  Abstractions over some aritmetic operations.
 */

namespace amgcl {
namespace math {

template <class T, class Enable = void>
struct scalar {
    typedef T type;
};

template <class T, class Enable = void>
struct zero_impl {
    static T get() {
        return 0;
    }
};

template <class T>
T zero() {
    return zero_impl<T>::get();
}

template <class T, class Enable = void>
struct identity_impl {
    static T get() {
        return 1;
    }
};

template <class T>
T identity() {
    return identity_impl<T>::get();
}

template <class T, class Enable = void>
struct constant_impl {
    static T get(typename scalar<T>::type c) {
        return c;
    }
};

template <class T>
T constant(typename scalar<T>::type c) {
    return constant_impl<T>::get(c);
}

template <class T, class Enable = void>
struct adjoint_impl {
    typedef T result_type;

    static result_type get(const T &v) {
        return v;
    }
};

template <class T>
typename adjoint_impl<T>::result_type
adjoint(const T &v) {
    return adjoint_impl<T>::get(v);
}

template <class T, class Enable = void>
struct inverse_impl {
    static T get(const T &v) {
        return 1 / v;
    }
};

template <class T>
T inverse(const T &v) {
    return inverse_impl<T>::get(v);
}

template <class T, class Enable = void>
struct abs_impl {
    static typename scalar<T>::type
    get(const T &v) {
        return std::fabs(v);
    }
};

template <class T>
typename scalar<T>::type abs(const T &v) {
    return abs_impl<T>::get(v);
}

} // namespace math
} // namespace amgcl

#endif
