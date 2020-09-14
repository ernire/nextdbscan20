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

//float get_lowest_e(float const e, long const n_dim) {
//    // TODO find a less wasteful formula to maintain precision
//    if (n_dim <= 3) {
//        return e / sqrtf(3);
//    } else if (n_dim <= 8) {
//        return e / sqrtf(3.5);
//    } else if (n_dim <= 30) {
//        return e / sqrtf(4);
//    } else if (n_dim <= 80) {
//        return e / sqrtf(5);
//    } else {
//        return e / sqrtf(6);
//    }
//}


/*
    static unsigned long determine_data_boundaries(s_vec<float> &v_min_bounds, s_vec<float> &v_max_bounds,
            float *v_coords, long const n_dim, unsigned long const n_coords, float const lowest_e, nextMPI &mpi)
            noexcept {
        float max_limit = INT32_MIN;
        calc_bounds(&v_min_bounds[0], &v_max_bounds[0], v_coords, n_dim, n_coords);
        mpi.allReduceMin(v_min_bounds, v_min_bounds, n_dim);
        mpi.allReduceMax(v_max_bounds, v_max_bounds, n_dim);
//        #ifdef MPI_ON
//        MPI_Allreduce(MPI_IN_PLACE, &min_bounds[0], n_dim, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
//        MPI_Allreduce(MPI_IN_PLACE, &max_bounds[0], n_dim, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
//        #endif
#pragma omp parallel for reduction(max: max_limit)
for (auto d = 0; d < n_dim; d++) {
if (v_max_bounds[d] - v_min_bounds[d] > max_limit)
max_limit = v_max_bounds[d] - v_min_bounds[d];
}
return static_cast<unsigned long>(ceilf(logf(max_limit / lowest_e) / logf(2))) + 1;
}
 */

inline int cell_index(s_vec<float> &v_coords, s_vec<float> &v_min_bounds, int const i, int const n_dim,
        int const d, float const e_l) {
    return (int)((v_coords[(i*n_dim)+d] - v_min_bounds[d]) / e_l);
}

inline bool dist_leq(float const *coord1, float const *coord2, long const n_dim, float const e2) noexcept {
    float tmp = 0;
//    #pragma omp simd
    for (auto d = 0; d < n_dim; d++) {
        tmp += (coord1[d] - coord2[d]) * (coord1[d] - coord2[d]);
    }
    return tmp <= e2;
}

inline static bool is_in_reach(const float *min1, const float *max1, const float *min2, const float *max2,
        long const n_dim, float const e) noexcept {
    for (auto d = 0; d < n_dim; ++d) {
        if ((min2[d] > (max1[d] + e) || max2[d] < (min1[d] - e))) {
            return false;
        }
//            if ((min2[d] > (max1[d] + e) || min2[d] < (min1[d] - e)) &&
//                (min1[d] > (max2[d] + e) || min1[d] < (min2[d] - e)) &&
//                (max2[d] > (max1[d] + e) || max2[d] < (min1[d] - e)) &&
//                (max1[d] > (max2[d] + e) || max1[d] < (min2[d] - e))) {
//                return false;
//            }
    }
    return true;
}

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

    /*
    magma_util::measure_duration("Init cells: ", mpi.rank == 0, [&]() -> void {
        nc.init_leaf_cells();
//        nc.build_tree();
        if (mpi.n_nodes > 1) {
            // SENDA bounds til þess að útiloka cellur


            // mpi all to all með fixed buffer
            // n_coords + 1 er max
        }
    });
    magma_util::measure_duration("Process cells: ", mpi.rank == 0, [&]() -> void {
        nc.process();
    });
    auto cores = nc.count_cores();
    std::cout << "CORES: " << cores << std::endl;
    */

    /*
    // TODO MPI
    auto e_l = get_lowest_e(e, n_dim);
    // make bottom nc-tree level
    exa::sort(v_coord_id, 0, v_coord_id.size(), [&](auto const i1, auto const i2) -> bool {
        for (auto d = 0; d < n_dim; ++d) {
            auto ci1 = cell_index(v_coords, v_min_bounds, i1, n_dim, d, e_l);
            auto ci2 = cell_index(v_coords, v_min_bounds, i2, n_dim, d, e_l);
            if (ci1 < ci2) {
                return true;
            } else if (ci1 > ci2) {
                return false;
            }
        }
        return false;
    });

    // Test lookup begin



    // Test lookup end

    s_vec<int> v_cell_size;
    s_vec<int> v_cell_offset(1, 0);
    s_vec<int> v_iota(v_coord_id.size());
    exa::iota(v_iota, 0, v_iota.size(), 0);
    exa::copy_if(v_iota, v_cell_offset, 1, v_iota.size(), 1, [&](auto const &i) -> bool {
        for (auto d = 0; d < n_dim; ++d) {
            auto ci1 = cell_index(v_coords, v_min_bounds, v_coord_id[i-1], n_dim, d, e_l);
            auto ci2 = cell_index(v_coords, v_min_bounds, v_coord_id[i], n_dim, d, e_l);
            if (ci1 != ci2)
                return true;
        }
        return false;
    });
    v_cell_size.resize(v_cell_offset.size());
    exa::iota(v_cell_size, 0, v_cell_size.size(), 0);
    exa::transform(v_cell_size, v_cell_size, 0, v_cell_size.size()-1, 0, [&](auto const &i) -> int {
       return  v_cell_offset[i+1] - v_cell_offset[i];
    });
    v_cell_size[v_cell_size.size()-1] = n_coord - v_cell_offset[v_cell_size.size()-1];
//    auto sum = exa::reduce(v_cell_size, 0, v_cell_size.size(), 0);
//    std::cout << "sum: " << sum << std::endl;

    std::cout << "Number of cells: " << v_cell_size.size() << std::endl;

    s_vec<int> v_cnt_point(n_coord, 0);
    auto v_cnt_cell = v_cell_size;
    auto e2 = e * e;
    */


    /*
    std::cout << "Calc cell maxmin" << std::endl;
    // Calculate max
    s_vec<float> v_cell_max(v_cell_offset.size()*n_dim);
    s_vec<float> v_cell_min(v_cell_offset.size()*n_dim);
    for (auto c = 0; c < v_cell_offset.size(); ++c) {
        for (auto i = 0; i < v_cell_size[c]; ++i) {
            for (auto d = 0; d < n_dim; ++d) {
                if (i == 0 || v_cell_max[c*n_dim+d] < v_coords[v_coord_id[v_cell_offset[c]+i]*n_dim+d]) {
                    v_cell_max[c*n_dim+d] = v_coords[v_coord_id[v_cell_offset[c]+i]*n_dim+d];
                }
                if (i == 0 || v_cell_min[c*n_dim+d] > v_coords[v_coord_id[v_cell_offset[c]+i]*n_dim+d]) {
                    v_cell_min[c*n_dim+d] = v_coords[v_coord_id[v_cell_offset[c]+i]*n_dim+d];
                }
            }
        }
    }

    std::cout << "Cell Bounds" << std::endl;

    s_vec<int> v_upper_bound(n_coord, 0);
    #pragma omp parallel for
    for (auto c1 = 0; c1 < v_cell_offset.size(); ++c1) {
        v_upper_bound[c1] = v_cell_offset.size();
        for (auto c2 = c1+1; c2 < v_cell_offset.size(); ++c2) {
            if (is_in_reach(&v_cell_min[c1*n_dim], &v_cell_max[c1*n_dim], &v_cell_min[c2*n_dim], &v_cell_max[c2*n_dim], n_dim, e)) {
                for (auto i = 0; i < v_cell_size[c1]; ++i) {
                    auto id1 = v_coord_id[v_cell_offset[c1]+i];
                    for (auto j = 0; j < v_cell_size[c2]; ++j) {
                        auto id2 = v_coord_id[v_cell_offset[c2]+j];
                        if (dist_leq(&v_coords[id1*n_dim], &v_coords[id2*n_dim], n_dim, e2)) {
#pragma omp atomic
                            ++v_cnt_point[id1];
#pragma omp atomic
                            ++v_cnt_point[id2];
                        }
                    }
                }
//                v_upper_bound[c1] = c2;
//                break;
            }
        }
    }
     */

    /*
    for (auto c1 = 0; c1 < v_cell_offset.size(); ++c1) {
        v_upper_bound[c1] = v_cell_offset.size();
        int ci1 = cell_index(v_coords, v_min_bounds, v_coord_id[v_cell_offset[c1]], n_dim, 0, e_l);
        for (auto c2 = c1+1; c2 < v_cell_offset.size(); ++c2) {
            int ci2 = cell_index(v_coords, v_min_bounds, v_coord_id[v_cell_offset[c2]], n_dim, 0, e_l);
            if (ci1 - ci2 > 2 || ci1 - ci2 < -2) {
                v_upper_bound[c1] = c2;
            }
//            if (ci1 != ci2) {
//                std::cout << "ci1: " << ci1 << " ci2: " << ci2 << std::endl;
//            }
//            if (ci1 - ci2 <= 100 && ci1 - ci2 >= -100) {
//                v_upper_bound[c1] = c2;
//                c2 = v_cell_offset.size();
//            }
        }
//        v_upper_bound[c1] = std::upper_bound(v_coord_id.begin(), v_coord_id.end(), );
    }
     */

    // Better solution
    /*
    #pragma omp parallel for
    for (auto c1 = 0; c1 < v_cell_offset.size(); ++c1) {
        for (auto c2 = c1+1; c2 < v_upper_bound[c1]; ++c2) {
            for (auto i = 0; i < v_cell_size[c1]; ++i) {
                auto id1 = v_coord_id[v_cell_offset[c1]+i];
                for (auto j = 0; j < v_cell_size[c2]; ++j) {
                    auto id2 = v_coord_id[v_cell_offset[c2]+j];
                    if (dist_leq(&v_coords[id1*n_dim], &v_coords[id2*n_dim], n_dim, e2)) {
                        #pragma omp atomic
                        ++v_cnt_point[id1];
                        #pragma omp atomic
                        ++v_cnt_point[id2];
                    }
                }
            }
        }
    }
     */




//    int cores = 0;
    /*
    for (auto c = 0; c < v_cell_offset.size(); ++c) {
        for (auto i = 0; i < v_cell_size[c]; ++i) {
            auto id = v_coord_id[v_cell_offset[c]+i];
            if (v_cnt_point[id] + v_cnt_cell[c] >= m) {
                ++cores;
            }
        }
    }
     */
//    std::cout << "CORES: " << cores << std::endl;


//    std::cout << "v_cell_begin: " << v_cell_offset.size() << " " << v_cell_offset[0] << " : " << v_cell_offset[1] << " : " << v_cell_offset[2] << std::endl;




//    min bounds: 0.0764974 0 0.0767574 0 8.36408e-05 0 0 0
//    max bounds: 0.833713 0.518469 0.900714 0.574745 0.423507 0.348579 0.407975 0.339591

    /*
    // double the input
    s_vec<int> v_dim_id;
    s_vec<int> v_coord_cell_index(v_coords.size() / n_dim);
//    s_vec<int> v_index_table(v_coord_cell_index.size());
//    v_index_table[0] = 1;

//            v_dim_index[(i*n_dim) + d] = static_cast<unsigned long>(
//                            floorf((v_coords[(v_begin_shortcut[i] * n_dim) + d] - v_min_bounds[d]) / e_lvl) + 1);
    d_vec<int> vv_unique_pos(n_dim);
    v_dim_id = v_coord_id;
    for (auto d = 0; d < n_dim; ++d) {
        // TODO test only
        std::fill(v_coord_cell_index.begin(), v_coord_cell_index.end(), -1);
        std::for_each(v_dim_id.begin(), v_dim_id.end(), [&](auto const &i) -> void {
            v_coord_cell_index[i] = (int)((v_coords[(i*n_dim)+d] - v_min_bounds[d]) / e_l);
        });
        // TODO remove check
        for (auto const &val : v_coord_cell_index) {
            assert(val != -1);
        }


        std::sort(v_dim_id.begin(), v_dim_id.end(), [&](auto const &i, auto const &j) -> bool {
           return  v_coord_cell_index[i] < v_coord_cell_index[j];
        });
        vv_unique_pos[d].resize(v_coord_id.size());
        vv_unique_pos[d][0] = 0;
        auto it = std::copy_if(std::next(v_coord_id.begin(), 1), v_coord_id.end(),
                std::next(vv_unique_pos[d].begin(), 1),
                [&](auto const &i) -> bool {
            return v_coord_cell_index[v_dim_id[i-1]] != v_coord_cell_index[v_dim_id[i]];
        });
        vv_unique_pos[d].resize(std::distance(vv_unique_pos[d].begin(), it));
        std::cout << "d: " << d << " unique size: " << vv_unique_pos[d].size() << std::endl;
//        v_cell_cnt_id_nz.resize(std::distance(v_cell_cnt_id_nz.begin(), it));

//        std::fill(std::next(v_index_table.begin(), 1), v_index_table.end(), 0);

//        std::for_each(std::next(v_dim_id.begin()), v_dim_id.end(), [&](auto const &i) -> void {

//        });
    }
     */
/*
    std::for_each(v_coord_id.begin(), v_coord_id.end(), [&](auto const &i) -> void {
        for (std::size_t d = 0; d < n_dim; ++d) {
            v_coord_cell_index[(i*n_dim)+d] = (int)((v_coords[(i*n_dim)+d] - v_min_bounds[d]) / e_l);
        }
    });


    s_vec<int> v_cell_size;
    s_vec<int> v_cell_offset;
//    s_vec<int> v_cell_cnt_id;
//    s_vec<int> v_cell_cnt_id_nz;
    for (std::size_t d = 0; d < n_dim; ++d) {
        v_cell_size.resize((v_max_bounds[d] - v_min_bounds[d]) / e_l + 1);
        std::fill(v_cell_size.begin(), v_cell_size.end(), 0);
        std::for_each(v_coord_id.begin(), v_coord_id.end(), [&](auto const &i) -> void {
            ++v_cell_size[v_coord_cell_index[(i*n_dim)+d]];
        });
        v_cell_offset = v_cell_size;
        std::exclusive_scan(v_cell_offset.begin(), v_cell_offset.end(), v_cell_offset.begin(), 0);
        v_cell_cnt_id.resize(v_cell_cnt.size());
        v_cell_cnt_id_nz.resize(v_cell_cnt.size());
        std::iota(v_cell_cnt_id.begin(), v_cell_cnt_id.end(), 0);
        std::for_each(v_coord_id.begin(), v_coord_id.end(), [&](auto const &i) -> void {
            ++v_cell_cnt[v_coord_cell_index[(i*n_dim)+d]];
        });
        std::cout << "d: " << d << " v_cell_cnt size: " << v_cell_cnt.size() << std::endl;
        auto it = std::copy_if(v_cell_cnt_id.begin(), v_cell_cnt_id.end(), v_cell_cnt_id_nz.begin(),
            [&](auto const i) -> bool {
                return v_cell_cnt[i] > 0;
            });
        v_cell_cnt_id_nz.resize(std::distance(v_cell_cnt_id_nz.begin(), it));
    }
                  */

    auto time_end = std::chrono::high_resolution_clock::now();
    std::cout << std::endl;
    std::cout << "Total Execution Time (without I/O): "
              << std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count()
              << " milliseconds\n";

    return nextdbscan::result();
}
