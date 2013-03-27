#include <iostream>
#include <cstdlib>
#include <utility>

#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <unsupported/Eigen/SparseExtra>

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

    // Read matrix in general format.
    Eigen::SparseMatrix<real, Eigen::RowMajor, int> GenA;
    Eigen::loadMarket(GenA, argv[1]);

    // Convert it to block format;
    int n = GenA.rows() / 4;
    std::vector<int>    row(n + 1, 0);
    std::vector<int>    col;
    std::vector<mblock> val;

    {
        std::vector<int> marker(n, -1);
        for(int i = 0; i < GenA.rows(); ++i) {
            int ii = i / 4;
            for(int j = GenA.outerIndexPtr()[i], e = GenA.outerIndexPtr()[i + 1]; j < e; ++j) {
                int jj = GenA.innerIndexPtr()[j] / 4;
                if (marker[jj] != ii) {
                    marker[jj] = ii;
                    row[ii + 1]++;
                }
            }
        }

        std::partial_sum(row.begin(), row.end(), row.begin());
        std::fill(marker.begin(), marker.end(), -1);

        col.resize(row.back());
        val.resize(row.back());
        for(int i = 0; i < GenA.rows(); ++i) {
            int gi = i / 4;
            int li = i % 4;
            int row_head = row[gi];
            int row_tail = row_tail;
            for(int j = GenA.outerIndexPtr()[i], e = GenA.outerIndexPtr()[i + 1]; j < e; ++j) {
                int gj = GenA.innerIndexPtr()[j] / 4;
                int lj = GenA.innerIndexPtr()[j] % 4;

                if (marker[gj] < row_head) {
                    marker[gj] = row_tail;

                    col[row_tail] = gj;
                    val[row_tail](li,lj) = GenA.valuePtr()[j];

                    row_tail++;
                } else {
                    val[marker[gj]](li,lj) = GenA.valuePtr()[j];
                }
            }
        }

    }

    EigenVector rhs(n), x(n);
    rhs.resize(n);
    for(int i = 0; i < n; ++i)
        for(int j = 0; j < 4; j++) {
            rhs[i][j] = 1;
            x[i][j] = 0;
        }

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
    prm.level.maxiter = 1000;
    prm.level.kcycle = 1;

    prof.tic("setup");
    AMG amg( amgcl::sparse::map(A), prm );
    prof.toc("setup");

    std::cout << amg << std::endl;

    // Solve the problem with CG method. Use AMG as a preconditioner:
    prof.tic("solve (cg)");
    //std::pair<int,double> cnv = amgcl::solve(A, rhs, amg, x, amgcl::cg_tag());
    std::pair<int,double> cnv = amg.solve(rhs, x);
    prof.toc("solve (cg)");

    std::cout << "Iterations: " << cnv.first  << std::endl
              << "Error:      " << cnv.second << std::endl
              << std::endl;

    std::cout << prof;
}
