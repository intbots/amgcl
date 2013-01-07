#include <iostream>

#include <Eigen/SparseCore>
#include <unsupported/Eigen/SparseExtra>

#include <vexcl/vexcl.hpp>

#include <amgcl/amgcl.hpp>
#include <amgcl/interp_aggr.hpp>
#include <amgcl/aggr_plain.hpp>
#include <amgcl/level_vexcl.hpp>
#include <amgcl/operations_eigen.hpp>
#include <amgcl/bicgstab.hpp>
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

    // Initialize VexCL context.
    vex::Context ctx( vex::Filter::Env && vex::Filter::DoublePrecision );
    if (!ctx.size()) {
        std::cerr << "No GPUs" << std::endl;
        return 1;
    }
    std::cout << ctx << std::endl;

    typedef amgcl::solver<
        real, int,
        amgcl::interp::aggregation<amgcl::aggr::plain>,
        amgcl::level::vexcl<amgcl::relax::spai0>
        > AMG;
    AMG::params prm;

    prm.interp.eps_strong   = 0; // Consider all connections as strong.
    prm.interp.dof_per_node = 4;

    prm.level.ctx   = &ctx;
    prm.level.npre  = 1;
    prm.level.npost = 2;

    prof.tic("setup");
    AMG amg(amgcl::sparse::map(A), prm);
    prof.toc("setup");

    std::cout << amg << std::endl;

    // Copy matrix to GPU(s).
    vex::SpMat<real, int, int> Agpu(ctx.queue(),
            A.rows(), A.cols(),
            A.outerIndexPtr(), A.innerIndexPtr(), A.valuePtr()
            );
    vex::vector<real> f(ctx.queue(), A.rows());
    vex::vector<real> x(ctx.queue(), A.rows());

    f = 1;
    x = 0;

    prof.tic("solve");
    std::pair<int,real> cnv = amgcl::solve(Agpu, f, amg, x, amgcl::bicg_tag());
    prof.toc("solve");

    std::cout << "Iterations: " << cnv.first  << std::endl
              << "Error:      " << cnv.second << std::endl
              << std::endl;

    std::cout << prof;
}
