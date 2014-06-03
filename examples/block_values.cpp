#include <iostream>
#include <fstream>
#include <iterator>
#include <algorithm>

#include <amgcl/amgcl.hpp>

#include <amgcl/math/eigen.hpp>
#include <amgcl/backend/builtin.hpp>
#include <amgcl/coarsening/aggregation.hpp>
#include <amgcl/relaxation/damped_jacobi.hpp>
#include <amgcl/solver/bicgstab.hpp>
#include <amgcl/profiler.hpp>

namespace amgcl {
    profiler<> prof("v2");
}

const size_t block_size = 4;
typedef Eigen::Matrix<double, block_size, block_size> M;
typedef Eigen::Matrix<double, block_size, 1         > V;

int main() {
    using amgcl::prof;

    typedef amgcl::amg<
        amgcl::backend::builtin<M, V>,
        amgcl::coarsening::aggregation,
        amgcl::relaxation::damped_jacobi
        > AMG;

    size_t nb;
    amgcl::backend::crs<M, long> A;
    std::vector<V> rhs;

    prof.tic("read");
    {
        std::vector<long>   sptr;
        std::vector<long>   scol;
        std::vector<double> sval;
        std::vector<double> srhs;

        std::istream_iterator<long>   iend;
        std::istream_iterator<double> dend;

        std::ifstream fptr("rows.txt");
        std::istream_iterator<long> iptr(fptr);

        std::ifstream fcol("cols.txt");
        std::istream_iterator<long> icol(fcol);

        std::ifstream fval("values.txt");
        std::istream_iterator<double> ival(fval);

        std::ifstream frhs("rhs.txt");
        std::istream_iterator<double> irhs(frhs);

        sptr.assign(iptr, iend);
        scol.assign(icol, iend);
        sval.assign(ival, dend);
        srhs.assign(irhs, dend);

        size_t n  = sptr.size() - 1;
        nb = n / block_size;

        A.nrows = nb;
        A.ncols = nb;
        A.ptr.resize(nb + 1, 0);

        std::vector<long> marker(nb, -1);

        // Count number of nonzeros in block matrix.
        for(size_t ib = 0, ia = ib * block_size; ib < nb; ++ib) {
            for(size_t k = 0; k < block_size && ia < n; ++k, ++ia) {
                for(long ja = sptr[ia]; ja < sptr[ia+1]; ++ja) {
                    long cb = scol[ja] / block_size;

                    if (marker[cb] != static_cast<long>(ib)) {
                        marker[cb]  = static_cast<long>(ib);
                        ++A.ptr[ib + 1];
                    }
                }
            }
        }

        std::fill(marker.begin(), marker.end(), -1);

        std::partial_sum(A.ptr.begin(), A.ptr.end(), A.ptr.begin());
        A.col.resize(A.ptr.back());
        A.val.resize(A.ptr.back(), M::Zero());

        // Fill the block matrix.
        for(size_t ib = 0, ia = ib * block_size; ib < nb; ++ib) {
            long row_beg = A.ptr[ib];
            long row_end = row_beg;

            for(size_t k = 0; k < block_size && ia < n; ++k, ++ia) {
                for(long ja = sptr[ia]; ja < sptr[ia+1]; ++ja) {
                    long   cb = scol[ja] / block_size;
                    long   cc = scol[ja] % block_size;
                    double va = sval[ja];

                    if (marker[cb] < row_beg) {
                        marker[cb] = row_end;
                        A.col[row_end] = cb;
                        A.val[row_end](k,cc) = va;
                        ++row_end;
                    } else {
                        A.val[marker[cb]](k,cc) = va;
                    }
                }
            }
        }

        rhs.resize(nb);
        for (size_t i = 0; i < nb; ++i)
            for(size_t k = 0; k < block_size; ++k)
                rhs[i](k) = srhs[block_size * i + k];
    }
    prof.toc("read");

    prof.tic("build");
    AMG::params prm;
    prm.coarsening.eps_strong = 0;
    AMG amg(A, prm);
    prof.toc("build");

    std::cout << amg << std::endl;

    std::vector<V> x(nb, V::Zero());

    amgcl::solver::bicgstab<AMG::backend_type> solve(nb);

    prof.tic("solve");
    size_t iters;
    double resid;
    boost::tie(iters, resid) = solve(amg.top_matrix(), rhs, amg, x);
    prof.toc("solve");

    std::cout << "Iterations: " << iters << std::endl
              << "Error:      " << resid << std::endl
              << std::endl;

    std::cout << amgcl::prof << std::endl;
}
