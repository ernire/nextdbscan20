//
// Created by Ernir Erlingsson on 16.8.2020.
//

#include <iostream>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <chrono>
#include <omp.h>
#include <cmath>
#include "nextdbscan.h"
#include "magma_input.h"
#include "nc_tree.h"

nextdbscan::result nextdbscan::start(int const m, float const e, int const n_thread, std::string const &in_file,
        magmaMPI mpi) noexcept {

    if (mpi.rank == 0) {
        std::cout << "Total of " << (n_thread * mpi.n_nodes) << " cores used on " << mpi.n_nodes << " node(s)." << std::endl;
    }

    s_vec<float> v_coord;
    int n_coord = -1, n_dim = -1;
    magma_util::measure_duration("Read Input Data: ", mpi.rank == 0, [&]() -> void {
        magma_input::read_input(in_file, v_coord, n_coord, n_dim, mpi.n_nodes, mpi.rank);
    });
    if (mpi.rank == 0) {
        std::cout << "Read " << n_coord << " points with " << n_dim << " dimensions. " << std::endl;
    }

    auto time_start = std::chrono::high_resolution_clock::now();

    nc_tree nc(v_coord, m, e, n_dim);
    magma_util::measure_duration("Determine Data Boundaries: ", mpi.rank == 0, [&]() -> void {
        nc.determine_data_bounds();
        if (mpi.n_nodes > 1) {
            mpi.allReduce(nc.v_min_bounds, nc.v_min_bounds, n_dim, magmaMPI::min);
            mpi.allReduce(nc.v_max_bounds, nc.v_max_bounds, n_dim, magmaMPI::max);
        }
        magma_util::print_vector("min bounds: " , nc.v_min_bounds);
        magma_util::print_vector("max bounds: " , nc.v_max_bounds);
    });
    nc.process6();

    auto time_end = std::chrono::high_resolution_clock::now();
    std::cout << std::endl;
    std::cout << "Total Execution Time (without I/O): "
              << std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count()
              << " milliseconds\n";

    return nextdbscan::result();
}
