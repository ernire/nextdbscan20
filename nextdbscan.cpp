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

    h_vec<float> v_coord;
    int n_coord = -1, n_dim = -1;
    magma_util::measure_duration("Read Input Data: ", mpi.rank == 0, [&]() -> void {
        magma_input::read_input(in_file, v_coord, n_coord, n_dim, mpi.n_nodes, mpi.rank);
    });
    if (mpi.rank == 0) {
        std::cout << "Read " << n_coord << " points with " << n_dim << " dimensions. " << std::endl;
    }

    auto time_start = std::chrono::high_resolution_clock::now();
    data_process nc(v_coord, m, e, n_dim);

    magma_util::measure_duration("Determine Data Boundaries: ", mpi.rank == 0, [&]() -> void {
        nc.determine_data_bounds();
        if (mpi.n_nodes > 1) {
            mpi.allReduce(nc.v_min_bounds, magmaMPI::min);
            mpi.allReduce(nc.v_max_bounds, magmaMPI::max);
        }
    });
#ifdef DEBUG_ON
    h_vec<float> v_min_bounds = nc.v_min_bounds;
    h_vec<float> v_max_bounds = nc.v_max_bounds;
    h_vec<int> v_dim_order = nc.v_dim_order;
    if (mpi.rank == 0) {
        magma_util::print_v("min bounds: ", &v_min_bounds[0], v_min_bounds.size());
        magma_util::print_v("max bounds: ", &v_max_bounds[0], v_max_bounds.size());
        magma_util::print_v("dim order: ", &v_dim_order[0], v_dim_order.size());
    }
#endif
    magma_util::measure_duration("Initialize Cells: ", mpi.rank == 0, [&]() -> void {
        nc.initialize_cells();
    });
#ifdef DEBUG_ON
    if (mpi.rank == 0)
        std::cout << "number of cells: " << nc.v_coord_cell_size.size() << std::endl;
    h_vec<int> v_dim_part_size = nc.v_dim_part_size;
    if (mpi.rank == 0) {
        magma_util::print_v("dim part size: ", &v_dim_part_size[0], v_dim_part_size.size());
    }
#endif
    magma_util::measure_duration("Process Points: ", mpi.rank == 0, [&]() -> void {
        nc.select_and_process(mpi);
    });
    auto time_end = std::chrono::high_resolution_clock::now();
    if (mpi.rank == 0) {
        std::cout << std::endl;
        std::cout << "Total Execution Time (excluding I/O): "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count()
                  << " milliseconds\n";
    }
    auto result = nextdbscan::result();
    magma_util::measure_duration("Collect Results: ", mpi.rank == 0, [&]() -> void {
        nc.get_result_meta(result.core_count, result.noise, result.clusters, result.n, mpi);
    });
    return result;
//    return nextdbscan::result();
}
