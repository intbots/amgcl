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
    prm.level.maxiter = 50;

    prof.tic("setup");
    AMG amg( amgcl::sparse::map(A), prm );
    prof.toc("setup");

    std::cout << amg << std::endl;

    // Solve the problem with CG method. Use AMG as a preconditioner:
    prof.tic("solve (cg)");
    std::pair<int,double> cnv = amg.solve(rhs, x);
    prof.toc("solve (cg)");

    std::cout << "Iterations: " << cnv.first  << std::endl
              << "Error:      " << cnv.second << std::endl
              << std::endl;

    std::cout << prof;
}
