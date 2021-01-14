//
// Created by Ernir Erlingsson on 16.8.2020.
//

#include <iostream>
#include <chrono>
#include "nextdbscan.h"
#include "magma_input.h"
#include "magma_util.h"


nextdbscan::result nextdbscan::start(int const m, float const e, int const n_thread, std::string const &in_file,
        magmaMPI mpi) noexcept {

    if (mpi.rank == 0) {
        std::cout << "Total of " << (n_thread * mpi.n_nodes) << " cores used on " << mpi.n_nodes << " node(s)." << std::endl;
    }

    auto start_timestamp = std::chrono::high_resolution_clock::now();

    h_vec<float> v_coord;
    int n_total_coord = -1, n_dim = -1;
    magma_util::measure_duration("Read Input Data: ", mpi.rank == 0, [&]() -> void {
        magma_input::read_input(in_file, v_coord, n_total_coord, n_dim, mpi.n_nodes, mpi.rank);
    });
    if (mpi.rank == 0) {
        std::cout << "Read " << n_total_coord << " aggregated points with " << n_dim << " dimensions. " << std::endl;
    }
    auto start_timestamp_no_io = std::chrono::high_resolution_clock::now();
    data_process dp(v_coord, m, e, n_dim, n_total_coord);

    magma_util::measure_duration("Determine Data Boundaries: ", mpi.rank == 0, [&]() -> void {
        dp.determine_data_bounds();
        if (mpi.n_nodes > 1) {
            mpi.allReduce(dp.v_min_bounds, magmaMPI::min);
            mpi.allReduce(dp.v_max_bounds, magmaMPI::max);
        }
    });
#ifdef DEBUG_ON
    h_vec<float> v_min_bounds = dp.v_min_bounds;
    h_vec<float> v_max_bounds = dp.v_max_bounds;
    h_vec<int> v_dim_order = dp.v_dim_order;
    if (mpi.rank == 0) {
        magma_util::print_v("min bounds: ", &v_min_bounds[0], v_min_bounds.size());
        magma_util::print_v("max bounds: ", &v_max_bounds[0], v_max_bounds.size());
        magma_util::print_v("dim order: ", &v_dim_order[0], v_dim_order.size());
    }
#endif

    magma_util::measure_duration("Build NC Tree: ", mpi.rank == 0, [&]() -> void {
        dp.build_nc_tree();
    });

    magma_util::measure_duration("Process NC Tree: ", mpi.rank == 0, [&]() -> void {
        dp.select_and_process(mpi);
    });

    auto result = nextdbscan::result();
    magma_util::measure_duration("Collect Results: ", mpi.rank == 0, [&]() -> void {
        dp.get_result_meta(result.processed, result.core_count, result.noise, result.clusters, result.n, mpi);
    });
    auto end_timestamp = std::chrono::high_resolution_clock::now();
    auto total_dur = std::chrono::duration_cast<std::chrono::milliseconds>(end_timestamp - start_timestamp).count();
    auto total_dur_no_io = std::chrono::duration_cast<std::chrono::milliseconds>(end_timestamp - start_timestamp_no_io).count();
    if (mpi.rank == 0) {
        std::cout << "Total Execution Time: " << total_dur << " milliseconds" << std::endl;
        std::cout << "Total Execution Time (without I/O): " << total_dur_no_io << " milliseconds" << std::endl;
    }
    return result;
}
