#define BOOST_TEST_MODULE TestNonScalar
#include <iostream>
#include <boost/test/unit_test.hpp>
#include <amgcl/math/eigen.hpp>
#include <amgcl/backend/builtin.hpp>
#include <amgcl/solver/cg.hpp>

typedef Eigen::Matrix<double, 2, 2>       matrix_scalar;
typedef Eigen::Matrix<double, 2, 1>       vector_scalar;

typedef amgcl::backend::builtin<
            matrix_scalar,
            vector_scalar
        > Backend;

typedef typename Backend::index_type      col_type;
typedef typename Backend::matrix          matrix;
typedef typename Backend::vector          vector;

matrix_scalar block() {
    matrix_scalar m;
    m << 2, -1,
        -1,  2;
    return m;
}

vector_scalar one (1.0, 1.0);
vector_scalar zero(0.0, 0.0);

BOOST_AUTO_TEST_SUITE( test_non_scalar_value_type )

BOOST_AUTO_TEST_CASE(test_clear)
{
    vector x(2, one);

    amgcl::backend::clear(x);

    for(int k = 0; k < 2; ++k)
        BOOST_CHECK_EQUAL(x[k], zero);

}

BOOST_AUTO_TEST_CASE(test_copy)
{
    vector x(2, one);
    vector y(2, zero);

    amgcl::backend::copy(x, y);

    for(int k = 0; k < 2; ++k)
        BOOST_CHECK_EQUAL(y[k], one);

}

BOOST_AUTO_TEST_CASE(test_inner_product)
{
    vector x(2, one);

    double v = amgcl::backend::inner_product(x, x);

    BOOST_CHECK_EQUAL(v, 4);

}

BOOST_AUTO_TEST_CASE(test_norm)
{
    vector x(2, one);

    double v = amgcl::backend::norm(x);

    BOOST_CHECK_EQUAL(v, 2);

}

BOOST_AUTO_TEST_CASE(test_axpby)
{
    vector x(2, one);
    vector y(2, one);

    amgcl::backend::axpby(1, x, -1, y);

    for(int k = 0; k < 2; ++k)
        BOOST_CHECK_EQUAL(y[k], zero);

}

BOOST_AUTO_TEST_CASE(test_vmul)
{
    std::vector<matrix_scalar> x(2, amgcl::math::one<matrix_scalar>());
    vector y(2, one);
    vector z(2, one);

    amgcl::backend::vmul(1, x, y, 1, z);

    for(int k = 0; k < 2; ++k)
        BOOST_CHECK_EQUAL(z[k], 2 * one);

}

BOOST_AUTO_TEST_CASE(test_spmv)
{
    col_type      ptr[] = {0, 1, 2};
    col_type      col[] = {0, 1};
    matrix_scalar val[] = {block(), block()};

    matrix A(2, 2,
             boost::make_iterator_range(ptr, ptr + 3),
             boost::make_iterator_range(col, col + 2),
             boost::make_iterator_range(val, val + 2)
             );

    vector x(2, one);
    vector y(2, zero);

    amgcl::backend::spmv(1, A, x, 0, y);

    for(int k = 0; k < 2; ++k)
        BOOST_CHECK_EQUAL(y[k], one);

}

BOOST_AUTO_TEST_CASE(test_residual)
{
    col_type      ptr[] = {0, 1, 2};
    col_type      col[] = {0, 1};
    matrix_scalar val[] = {block(), block()};

    matrix A(2, 2,
             boost::make_iterator_range(ptr, ptr + 3),
             boost::make_iterator_range(col, col + 2),
             boost::make_iterator_range(val, val + 2)
             );

    vector x(2, one);
    vector y(2, one);

    amgcl::backend::residual(y, A, x, y);

    for(int k = 0; k < 2; ++k)
        BOOST_CHECK_EQUAL(y[k], zero);

}

struct dummy_preconditioner {
    template <class V>
    void operator()(const V &f, V &x) const {
        amgcl::backend::copy(f, x);
    }
};

BOOST_AUTO_TEST_CASE(test_cg)
{
    col_type      ptr[] = {0, 1, 2};
    col_type      col[] = {0, 1};
    matrix_scalar val[] = {block(), block()};

    matrix A(2, 2,
             boost::make_iterator_range(ptr, ptr + 3),
             boost::make_iterator_range(col, col + 2),
             boost::make_iterator_range(val, val + 2)
             );

    vector x(2, zero);
    vector y(2, one);

    amgcl::solver::cg< Backend > solve(2);
    solve(A, y, dummy_preconditioner(), x);

    std::cout
        << x[0] << std::endl
        << x[1] << std::endl;
}

BOOST_AUTO_TEST_SUITE_END()
