#include <iostream>

#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <unsupported/Eigen/SparseExtra>

#include <amgcl/amgcl.hpp>
#include <amgcl/interp_aggr.hpp>
#include <amgcl/aggr_plain.hpp>
#include <amgcl/level_cpu.hpp>
#include <amgcl/operations_eigen.hpp>
#include <amgcl/gmres.hpp>
#include <amgcl/profiler.hpp>

typedef double real;
typedef Eigen::SparseMatrix<real, Eigen::RowMajor, int> EigenMatrix;
typedef Eigen::Matrix<real, Eigen::Dynamic, 1> EigenVector;

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " matrix.mm" << std::endl;
        return 1;
    }

    amgcl::profiler<> prof(argv[0]);

    prof.tic("read problem");
    EigenMatrix A;
    Eigen::loadMarket(A, argv[1]);
    prof.toc("read problem");

    std::cout << "Matrix: " << A.rows() << "x" << A.cols() << std::endl;

    typedef amgcl::solver<
        real, int,
        amgcl::interp::aggregation<amgcl::aggr::plain>,
        amgcl::level::cpu<amgcl::relax::ilu>
        > AMG;
    AMG::params prm;

    prof.tic("setup");
    AMG amg(amgcl::sparse::map(A), prm);
    prof.toc("setup");

    std::cout << amg << std::endl;

    EigenVector f = EigenVector::Ones(A.rows());
    EigenVector x = EigenVector::Zero(A.rows());

    prof.tic("solve");
    std::pair<int,real> cnv = amgcl::solve(A, f, amg, x, amgcl::gmres_tag<30>());
    prof.toc("solve");

    std::cout << "Iterations: " << cnv.first  << std::endl
              << "Error:      " << cnv.second << std::endl
              << std::endl;

    std::cout << prof;
}
