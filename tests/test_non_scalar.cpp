#define BOOST_TEST_MODULE TestNonScalar
#include <iostream>
#include <boost/test/unit_test.hpp>
#include <amgcl/amgcl.hpp>
#include <amgcl/math/eigen.hpp>
#include <amgcl/backend/builtin.hpp>
#include <amgcl/coarsening/aggregation.hpp>
#include <amgcl/relaxation/damped_jacobi.hpp>
#include <amgcl/solver/cg.hpp>
#include "sample_problem.hpp"

typedef Eigen::Matrix<double, 2, 2>       matrix_scalar;
typedef Eigen::Matrix<double, 2, 1>       vector_scalar;

typedef amgcl::backend::builtin<
            matrix_scalar,
            vector_scalar
        > Backend;

typedef typename Backend::index_type      col_type;
typedef typename Backend::matrix          matrix;
typedef std::vector<matrix_scalar>        diagonal_type;
typedef typename Backend::vector          vector;

matrix_scalar block() {
    matrix_scalar m;
    m << 2, -1,
        -1,  2;
    return m;
}

vector_scalar one  = amgcl::math::constant<vector_scalar>(1);
vector_scalar zero = amgcl::math::zero<vector_scalar>();

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
    diagonal_type x(2, amgcl::math::identity<matrix_scalar>());
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

    for(int k = 0; k < 2; ++k)
        BOOST_CHECK_EQUAL(x[k], one);
}

BOOST_AUTO_TEST_CASE(test_diagonal)
{
    col_type      ptr[] = {0, 1, 2};
    col_type      col[] = {0, 1};
    matrix_scalar val[] = {block(), block()};

    matrix A(2, 2,
             boost::make_iterator_range(ptr, ptr + 3),
             boost::make_iterator_range(col, col + 2),
             boost::make_iterator_range(val, val + 2)
             );

    diagonal_type dia = diagonal(A);
    diagonal_type ida = diagonal(A, true);

    for(int k = 0; k < 2; ++k) {
        BOOST_CHECK_EQUAL(dia[k], val[k]);
        BOOST_CHECK_EQUAL(dia[k] * ida[k], amgcl::math::identity<matrix_scalar>());
    }
}

BOOST_AUTO_TEST_CASE(test_inverse_n_product)
{
    col_type      ptr[] = {0, 1, 2};
    col_type      col[] = {0, 1};
    matrix_scalar val[] = {block(), block()};

    matrix A(2, 2,
             boost::make_iterator_range(ptr, ptr + 3),
             boost::make_iterator_range(col, col + 2),
             boost::make_iterator_range(val, val + 2)
             );

    matrix Ai = inverse(A);
    
    diagonal_type dia = diagonal(product(Ai, A));

    for(int k = 0; k < 2; ++k) {
        BOOST_CHECK_EQUAL(dia[k], amgcl::math::identity<matrix_scalar>());
    }
}

BOOST_AUTO_TEST_CASE(test_coarsening)
{
    col_type      ptr[] = {0, 1, 2};
    col_type      col[] = {0, 1};
    matrix_scalar val[] = {block(), block()};

    matrix A(2, 2,
             boost::make_iterator_range(ptr, ptr + 3),
             boost::make_iterator_range(col, col + 2),
             boost::make_iterator_range(val, val + 2)
             );

    typedef amgcl::coarsening::aggregation Coarsening;

    boost::shared_ptr<matrix> P;
    boost::shared_ptr<matrix> R;

    boost::tie(P, R) = Coarsening::transfer_operators(A, Coarsening::params());
}

BOOST_AUTO_TEST_CASE(full_test)
{
    matrix A;
    vector rhs;

    size_t n = A.nrows = A.ncols = sample_problem(32, A.val, A.col, A.ptr, rhs);

    typedef amgcl::amg<
        Backend,
        amgcl::coarsening::aggregation,
        amgcl::relaxation::damped_jacobi
        > AMG;

    AMG amg(A);
    std::cout << amg << std::endl;

    vector x(n, zero);
    amgcl::solver::cg<Backend> solve(n);

    size_t iters;
    double resid;
    boost::tie(iters, resid) = solve(amg.top_matrix(), rhs, amg, x);

    std::cout << "Iterations: " << iters << std::endl
              << "Error:      " << resid << std::endl
              << std::endl;
}

BOOST_AUTO_TEST_SUITE_END()
