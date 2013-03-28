#include <iostream>
#include <cstdlib>
#include <utility>

#include <Eigen/Dense>
#include <Eigen/SparseCore>

#include <amgcl/operations_eigen.hpp>
#include <amgcl/amgcl.hpp>
#include <amgcl/interp_smoothed_aggr.hpp>
#include <amgcl/aggr_plain.hpp>
#include <amgcl/level_cpu.hpp>
#include <amgcl/profiler.hpp>

#include "read.hpp"

typedef double real;
typedef Eigen::Matrix<real, 4, 4> mblock;
typedef Eigen::Matrix<real, 4, 1> vblock;

typedef Eigen::Matrix<vblock, Eigen::Dynamic, 1> EigenVector;

namespace amgcl {
    profiler<> prof("block");
}
using amgcl::prof;

double inner_prod(const EigenVector &x, const EigenVector &y) {
    double s = 0.0;
#pragma omp parallel for reduction(+:s) schedule(dynamic, 1024)
    for(size_t i = 0; i < x.size(); ++i) {
        s += amgcl::transpose(x[i]) * y[i];
    }
    return s;
}

double norm(const EigenVector &x) {
    return sqrt(inner_prod(x, x));
}

template <class matrix, class vector, class precond>
std::pair< int, double >
solve(const matrix &A, const vector &rhs, const precond &P, vector &x)
{
    TIC("solver");
    const size_t n = x.size();

    vector r(n), s(n), p(n), q(n);
#pragma omp parallel for schedule(dynamic, 1024)
    for(long i = 0; i < n; ++i) {
        size_t j = A.outerIndexPtr()[i];
        size_t e = A.outerIndexPtr()[i+1];
        vblock buf = rhs[i];
        for(; j < e; ++j) {
            buf -= A.valuePtr()[j] * x[A.innerIndexPtr()[j]];
        }
        r[i] = buf;
    }

    double rho1 = 0, rho2 = 0;
    double norm_of_rhs = norm(rhs);

    if (norm_of_rhs == 0) {
#pragma omp parallel for schedule(dynamic, 1024)
        for(long i = 0; i < n; ++i) x[i] = amgcl::zero<vblock>();
        return std::make_pair(0, norm_of_rhs);
    }

    int     iter = 0;
    double  res;

    for(; (res = norm(r) / norm_of_rhs) > 1e-8 && iter < 100; ++iter)
    {
#pragma omp parallel for schedule(dynamic, 1024)
        for(long i = 0; i < n; ++i)
            s[i] = amgcl::zero<vblock>();
        TOC("solver");
        P.apply(r, s);
        TIC("solver");

        rho2 = rho1;
        rho1 = inner_prod(r, s);

        if (iter) {
#pragma omp parallel for schedule(dynamic, 1024)
            for(long i = 0; i < n; ++i)
                p[i] = s[i] + (rho1 / rho2) * p[i];
        } else {
            p = s;
        }

#pragma omp parallel for schedule(dynamic, 1024)
        for(long i = 0; i < n; ++i) {
            size_t j = A.outerIndexPtr()[i];
            size_t e = A.outerIndexPtr()[i+1];
            vblock buf = amgcl::zero<vblock>();
            for(; j < e; ++j) {
                buf += A.valuePtr()[j] * p[A.innerIndexPtr()[j]];
            }
            q[i] = buf;
        }

        double alpha = rho1 / inner_prod(q, p);

#pragma omp parallel for schedule(dynamic, 1024)
        for(long i = 0; i < n; ++i) {
            x[i] += alpha * p[i];
            r[i] -= alpha * q[i];
        }
    }

    TOC("solver");
    return std::make_pair(iter, res);
}
int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <problem.dat>" << std::endl;
        return 1;
    }

    // Read matrix and rhs from a binary file.
    std::vector<int>  row;
    std::vector<int>  col;
    std::vector<real> gen_val;
    std::vector<real> gen_rhs;
    int n = read_problem(argv[1], row, col, gen_val, gen_rhs);

    // Convert the problem to block format.
    std::vector<mblock> val(row.back());
    for(int i = 0; i < row.back(); i++)
        val[i] = mblock::Identity() * gen_val[i];

    EigenVector x(n), rhs(n);
    for(int i = 0; i < n; i++) {
        x[i]   = vblock::Constant(0);
        rhs[i] = vblock::Constant(gen_rhs[i]);
    }

    // Map the block matrix.
    Eigen::MappedSparseMatrix<mblock, Eigen::RowMajor, int> A(
            n, n, row.back(), row.data(), col.data(), val.data()
            );

    // Build the preconditioner:
    typedef amgcl::solver<
        mblock, vblock, int,
        amgcl::interp::smoothed_aggregation<amgcl::aggr::plain>,
        amgcl::level::cpu<amgcl::relax::spai0>
        > AMG;

    AMG::params prm;
    prm.level.npre    = 2;
    prm.level.npost   = 2;
    prm.level.maxiter = 50;

    prof.tic("setup");
    AMG amg( amgcl::sparse::map(A), prm );
    prof.toc("setup");

    std::cout << amg << std::endl;

    // Solve the problem with CG method. Use AMG as a preconditioner:
    prof.tic("solve (cg)");
    std::pair<int,double> cnv = solve(A, rhs, amg, x);
    prof.toc("solve (cg)");

    std::cout << "Iterations: " << cnv.first  << std::endl
              << "Error:      " << cnv.second << std::endl
              << std::endl;

    int center = 0.5 * (sqrt(n) +  n);
    std::cout << x[center] << std::endl;

    std::cout << prof;
}
