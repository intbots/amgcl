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
#include <amgcl/cg.hpp>
#include <amgcl/profiler.hpp>

#include "read.hpp"

typedef double real;
typedef Eigen::Matrix<real, 4, 4> mblock;
typedef Eigen::Matrix<real, 4, 1> vblock;

typedef Eigen::Matrix<vblock, Eigen::Dynamic, 1> EigenVector;

int main(int argc, char *argv[]) {
    amgcl::profiler<> prof(argv[0]);

    // Read matrix and rhs from a binary file.
    std::vector<int>    row;
    std::vector<int>    col;
    std::vector<mblock> val;
    EigenVector         rhs;

    // TODO: Init matrix
    int n = 100;

    // Wrap the matrix into Eigen Map.
    Eigen::MappedSparseMatrix<mblock, Eigen::RowMajor, int> A(
            n, n, row.back(), row.data(), col.data(), val.data()
            );

    // Build the preconditioner:
    typedef amgcl::solver<
        mblock, vblock, int,
        amgcl::interp::smoothed_aggregation<amgcl::aggr::plain>,
        amgcl::level::cpu<amgcl::relax::spai0>
        > AMG;

    // Use K-Cycle on each level to improve convergence:
    AMG::params prm;
    prm.level.kcycle = 1;

    prof.tic("setup");
    AMG amg( amgcl::sparse::map(A), prm );
    prof.toc("setup");

    std::cout << amg << std::endl;

    // Solve the problem with CG method. Use AMG as a preconditioner:
    EigenVector x(n);// = EigenVector::Zero(n);
    prof.tic("solve (cg)");
    //std::pair<int,double> cnv = amgcl::solve(A, rhs, amg, x, amgcl::cg_tag());
    std::pair<int,double> cnv = amg.solve(rhs, x);
    prof.toc("solve (cg)");

    std::cout << "Iterations: " << cnv.first  << std::endl
              << "Error:      " << cnv.second << std::endl
              << std::endl;

    std::cout << prof;
}
