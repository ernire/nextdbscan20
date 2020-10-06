//
// Created by Ernir Erlingsson on 19.8.2020.
//

#include <iostream>
#include <unordered_map>
#include "magma_util.h"
#include "data_process.h"
#ifdef OMP_ON
#include "magma_exa_omp.h"
#elif CUDA_ON
#include "magma_exa_cu.h"
//#include "magma_exa_cu.cuh"
#else
#include "magma_exa.h"
#endif

void data_process::determine_data_bounds() noexcept {

    std::cout << "m: " << m << " e: " << e << std::endl;

    v_min_bounds.resize(n_dim);
    v_max_bounds.resize(n_dim);
    v_coord_id.resize(n_coord);
    exa::iota(v_coord_id, 0, v_coord_id.size(), 0);
    auto const _n_dim = n_dim;
    auto const it_coord = v_coord.begin();
    for (int d = 0; d < n_dim; ++d) {
        auto minmax = exa::minmax_element(v_coord_id, 0, v_coord_id.size(),
                [=]
#ifdef CUDA_ON
        __device__
#endif
                (auto const i1, auto const i2) -> bool {
                    return it_coord[i1 * _n_dim + d] < it_coord[i2 * _n_dim + d];
                });
        v_min_bounds[d] = v_coord[(minmax.first * n_dim) + d];
        v_max_bounds[d] = v_coord[(minmax.second * n_dim) + d];
    }
    v_dim_order.resize(n_dim);
    exa::iota(v_dim_order, 0, v_dim_order.size(), 0);
    auto const it_min = v_min_bounds.begin();
    auto const it_max = v_max_bounds.begin();
    exa::sort(v_dim_order, 0, v_dim_order.size(), [=]
#ifdef CUDA_ON
            __device__
#endif
        (int const &d1, int const &d2) -> bool {
       return (it_max[d1] - it_min[d1]) > (it_max[d2] - it_min[d2]);
    });
}

void data_process::collect_cells_in_reach(d_vec<int> &v_point_index, d_vec<int> &v_cell_reach,
        d_vec<int> &v_point_reach_offset, d_vec<int> &v_point_reach_size) noexcept {
    int const n_points = v_point_index.size();
    d_vec<int> v_point_reach_full(9 * n_points, -1);

    // Todo remove initialization
    d_vec<int> v_value(v_point_index.size());
    d_vec<int> v_lower_bound(v_point_index.size() * 3, -1);
    auto const _dim_part_0 = v_dim_part_size[0];
    auto const _dim_part_1 = v_dim_part_size[1];

    exa::transform(v_point_index, 0, v_point_index.size(), v_value, 0, []
#ifdef CUDA_ON
        __device__
#endif
    (auto const &v) -> auto {
        return v - 1;
    });
    exa::lower_bound(v_coord_cell_index, 0, v_coord_cell_index.size(), v_value, 0,
            v_value.size(), v_lower_bound, 0, 2);
    exa::transform(v_point_index, 0, v_point_index.size(), v_value, 0, [=]
#ifdef CUDA_ON
        __device__
#endif
    (auto const &v) -> auto {
        return v - _dim_part_0 - 1;
    });
    exa::lower_bound(v_coord_cell_index, 0, v_coord_cell_index.size(), v_value, 0,
            v_value.size(), v_lower_bound, 1, 2);
    exa::transform(v_point_index, 0, v_point_index.size(), v_value, 0, [=]
#ifdef CUDA_ON
        __device__
#endif
    (auto const &v) -> auto {
        return v + _dim_part_0 - 1;
    });
    exa::lower_bound(v_coord_cell_index, 0, v_coord_cell_index.size(), v_value, 0,
            v_value.size(), v_lower_bound, 2, 2);

    auto const it_point_reach_size = v_point_reach_size.begin();
    auto const it_point_index = v_point_index.begin();
    auto const it_full_reach = v_point_reach_full.begin();
    auto const it_bounds = v_lower_bound.begin();
    auto const it_coord_index = v_coord_cell_index.begin();

    exa::for_each(0, v_point_index.size(), [=]
#ifdef CUDA_ON
        __device__
#endif
    (auto const &i) -> void {
//        if (*(it_point_index + i) < 0) {
//            *(it_point_reach_size + i) = 0;
        if (it_point_index[i] < 0) {
            it_point_reach_size[i] = 0;
            return;
        }
        auto it_begin = it_full_reach + (i * 9);
        auto it_out = it_begin;
        int val = *(it_bounds + (i * 3));
        if (*(it_coord_index + val) == *(it_point_index + i) - 1) {
            *(it_out++) = val;
            ++val;
        }
        if (*(it_coord_index + val) == *(it_point_index + i)) {
            *(it_out++) = val;
            ++val;
        }
        if (*(it_coord_index + val) == *(it_point_index + i) + 1) {
            *(it_out++) = val;
            ++val;
        }
        val = *(it_bounds + (i * 3) + 1);
        if (*(it_coord_index + val) == *(it_point_index + i) - _dim_part_0 - 1) {
            *(it_out++) = val;
            ++val;
        }
        if (*(it_coord_index + val) == *(it_point_index + i) - _dim_part_0) {
            *(it_out++) = val;
            ++val;
        }
        if (*(it_coord_index + val) == *(it_point_index + i) - _dim_part_0 + 1) {
            *(it_out++) = val;
            ++val;
        }
        val = *(it_bounds + (i * 3) + 2);
        if (*(it_coord_index + val) == *(it_point_index + i) + _dim_part_0 - 1) {
            *(it_out++) = val;
            ++val;
        }
        if (*(it_coord_index + val) == *(it_point_index + i) + _dim_part_0) {
            *(it_out++) = val;
            ++val;
        }
        if (*(it_coord_index + val) == *(it_point_index + i) + _dim_part_0 + 1) {
            *(it_out++) = val;
            ++val;
        }
        *(it_point_reach_size + i) = it_out - it_begin;
    });

    exa::exclusive_scan(v_point_reach_size, v_point_reach_offset, 0, v_point_reach_size.size(), 0, 0);
    v_cell_reach.resize(v_point_reach_full.size());
    exa::copy_if(v_point_reach_full, 0, v_point_reach_full.size(), v_cell_reach, 0, []
#ifdef CUDA_ON
    __device__
#endif
    (int const &val) -> bool {
        return val >= 0;
    });
}

void data_process::process_points(d_vec<int> &v_point_id, d_vec<float> &v_point_data, magmaMPI mpi) noexcept {
    auto const it_coord_status = v_coord_status.begin();
    auto const it_point_id = v_point_id.begin();
    exa::for_each(0, v_point_id.size(), [=]
#ifdef CUDA_ON
        __device__
#endif
    (int const &i) -> void {
        if (*(it_point_id + i) >= 0) {
//        if (v_point_id[i] >= 0) {
            // local, we can modify the status
            it_coord_status[it_point_id[i]] = 1; // PROCESSED
//            *(it_coord_status + *(it_point_id + i)) = PROCESSED;
//            v_coord_status[v_point_id[i]] = PROCESSED;
        }
    });
    // calculate cell index
    d_vec<int> v_point_index(v_point_id.size());
//    magma_util::measure_duration("Point Index: ", mpi.rank == 0, [&]() -> void {
        index_points(v_point_data, v_point_index);
//    });

    // obtain reach
    d_vec<int> v_point_cells_in_reach(v_point_id.size());
    d_vec<int> v_point_cell_reach_offset(v_point_id.size());
    d_vec<int> v_point_cell_reach_size(v_point_id.size());

//    magma_util::measure_duration("Collect Cells: ", mpi.rank == 0, [&]() -> void {
        collect_cells_in_reach(v_point_index, v_point_cells_in_reach, v_point_cell_reach_offset,
                v_point_cell_reach_size);
//    });

#ifdef DEBUG_ON
    if (mpi.rank == 0)
        std::cout << "v_point_cells_in_reach size: " << v_point_cells_in_reach.size() << std::endl;
#endif

    d_vec<int> v_points_in_reach_size(v_point_id.size(), 0);
    d_vec<int> v_points_in_reach_offset(v_point_id.size());

    auto const it_point_cell_reach_size = v_point_cell_reach_size.begin();
    auto const it_coord_cell_size = v_coord_cell_size.begin();
    auto const it_point_cells_in_reach = v_point_cells_in_reach.begin();
    auto const it_point_cell_reach_offset = v_point_cell_reach_offset.begin();
    auto const it_points_in_reach_size = v_points_in_reach_size.begin();
    // calculate points in reach for each processed point
    exa::for_each(0, v_point_id.size(), [=]
#ifdef CUDA_ON
    __device__
#endif
    (int const &i) -> void {
        auto p_sum = 0;
        for (int j = 0; j < it_point_cell_reach_size[i]; ++j) {
            p_sum += it_coord_cell_size[it_point_cells_in_reach[it_point_cell_reach_offset[i] + j]];
        }
        it_points_in_reach_size[i] = p_sum;
//        for (int j = 0; j < it_point_cell_reach_size[i]; ++j) {
//            p_sum += v_coord_cell_size[v_point_cells_in_reach[v_point_cell_reach_offset[i] + j]];
//        }
//        v_points_in_reach_size[i] = p_sum;
    });

    auto const it_points_in_reach = v_points_in_reach_size.begin();
    d_vec<int> v_sorted_order(v_point_id.size());
    exa::iota(v_sorted_order, 0, v_sorted_order.size(), 0);
    exa::sort(v_sorted_order, 0, v_sorted_order.size(), [=]
#ifdef CUDA_ON
    __device__
#endif
    (auto const &i1, auto const &i2) -> bool {
        return it_points_in_reach[i1] > it_points_in_reach[i2];
//        return *(it_points_in_reach + i1) > *(it_points_in_reach + i2);
    });

    std::cout << "sorted order: " << v_sorted_order[0] << " : " << v_sorted_order[1] << " : " << v_sorted_order[2] << std::endl;

//    thrust::sequence(v_sorted_order.begin(), v_sorted_order.end(), 0);
//    thrust::sort(v_sorted_order.begin(), v_sorted_order.end(), [=]__device__(int const &i1, int const &i2) -> bool {
//            return *(it_points_in_reach + i1) > *(it_points_in_reach + i2);
//    });


    d_vec<int> v_point_nn(v_point_id.size(), 0);
    auto const it_coord_id = v_coord_id.begin();
    auto const it_coord_cell_offset = v_coord_cell_offset.begin();
    auto const it_point_data = v_point_data.begin();
    auto const it_coord_data = v_coord.begin();
    auto const it_point_nn = v_point_nn.begin();
    auto const it_coord_nn = v_coord_nn.begin();
    auto const it_sorted_order = v_sorted_order.begin();
    auto const _n_dim = n_dim;
    auto const _e2 = e2;
    auto const _m = m;
    exa::for_each(0, v_point_id.size(), [=]
#ifdef CUDA_ON
    __device__
#endif
    (auto const &j) -> void {
        int nn = 0;
        float tmp, result;
        int i = *(it_sorted_order + j);
        for (int ci = 0; ci < it_point_cell_reach_size[i]; ++ci) {
            int c = it_point_cells_in_reach[it_point_cell_reach_offset[i] + ci];
            for (int j = 0; j < it_coord_cell_size[c]; ++j) {
                int id2 = it_coord_id[it_coord_cell_offset[c] + j];
                result = 0;
                // TODO clean
                for (int d = 0; d < _n_dim; d++) {
                    tmp = *(it_point_data + (i * _n_dim) + d) - *(it_coord_data + (id2 * _n_dim) + d);
                    result += tmp * tmp;
                }
                if (result <= _e2) {
                    ++nn;
                }
            }
        }
        *(it_point_nn + i) = nn;
        if (it_point_nn[i] >= _m && it_point_id[i] >= 0) {
            it_coord_nn[it_point_id[i]] = it_point_nn[i];
        }
    });



    /*
    exa::for_each(0, v_point_id.size(), [&](int const &i) -> void {
       std::fill(std::next(v_hit_table_id_1.begin(), v_points_in_reach_offset[i]),
               std::next(v_hit_table_id_1.begin(), v_points_in_reach_offset[i] + v_points_in_reach_size[i]),
               i);
    });
    // Make cell offset and size
    s_vec<int> v_cell_reach_size(v_point_cells_in_reach.size());
    s_vec<int> v_cell_reach_offset(v_point_cells_in_reach.size());
    exa::transform(v_point_cells_in_reach, v_cell_reach_size, 0, v_point_cells_in_reach.size(), 0,
            [&](int const &c_id) -> int {
       return v_coord_cell_size[c_id];
    });
    exa::exclusive_scan(v_cell_reach_size, v_cell_reach_offset, 0, v_cell_reach_size.size(), 0, 0);
    exa::for_each(0, v_point_cells_in_reach.size(), [&](int const &i) -> void {
        for (int j = 0; j < v_cell_reach_size[i]; ++j) {
            v_hit_table_id_2[v_cell_reach_offset[i] + j] = v_coord_id[v_coord_cell_offset[v_point_cells_in_reach[i]] + j];
        }
    });
//    auto hit1 = exa::reduce(v_hit_table_id_1, 0, v_hit_table_id_1.size(), 0);
//    auto hit2 = exa::reduce(v_hit_table_id_2, 0, v_hit_table_id_2.size(), 0);
//    std::cout << "hit1: " << hit1 << " hit2: " << hit2 << std::endl;
    s_vec<int> v_point_nn(v_point_id.size(), 0);
    exa::for_each(0, v_hit_table_id_1.size(), [&](int const &i) -> void {
        if (!dist_leq(&v_point_data[v_hit_table_id_1[i]*n_dim], &v_coord[v_hit_table_id_2[i]*n_dim], n_dim, e2)) {
            v_hit_table_id_2[i] = -1;
        } else {
            ++v_point_nn[v_hit_table_id_1[i]];
//            if (v_coord_status[v_hit_table_id_2[i]] != PROCESSED) {
//                ++v_coord_nn[v_hit_table_id_2[i]];
//            }
        }
    });
         */
#ifdef MPI_ON
    mpi.allReduce(v_point_nn, magmaMPI::sum);
#endif

    /*
    d_vec<int> v_point_cluster(v_point_id.size(), NO_CLUSTER);
    d_vec<int> v_point_status(v_point_id.size(), 0);

    exa::for_each(0, v_point_id.size(), [&](int const &i) -> void {
        if (v_point_nn[i] >= m) {
            v_point_cluster[i] = i + cluster_size;
            if (v_point_id[i] >= 0) {
                v_coord_nn[v_point_id[i]] = v_point_nn[i];
                if (v_coord_cluster[v_point_id[i]] == NO_CLUSTER) {
                    v_coord_cluster[v_point_id[i]] = v_point_cluster[i];
                } else {
                    v_point_cluster[i] = v_coord_cluster[v_point_id[i]];
                }
            }
        }
    });
     */
    /*
    bool is_done = false;
    int iter_cnt = 0;
    while (!is_done) {
        assert(exa::reduce(v_point_status, 0, v_point_status.size(), 0) == 0);
        exa::for_each(0, v_point_id.size(), [&](int const &i) -> void {
            if (v_point_nn[i] >= m) {
                for (int j = 0; j < v_points_in_reach_size[i]; ++j) {
                    auto id2 = v_hit_table[v_points_in_reach_offset[i] + j];
                    if (id2 == -1) continue;
                    if (v_coord_nn[id2] >= m) {
                        if (v_coord_cluster[id2] == NO_CLUSTER) {
                            v_coord_cluster[v_point_id[i]] = v_point_cluster[i];
                        } else if (v_coord_cluster[id2] < v_point_cluster[i]) {
    //                            if (v_point_cluster[i] != i + cluster_size) {
    //                                std::cout << "CHECKPOINT!" << std::endl;
    //                            }
                            v_point_cluster[i] = v_coord_cluster[id2];
                            if (v_point_id[i] >= 0) {
                                v_coord_cluster[v_point_id[i]] = v_point_cluster[i];
                            }
                            v_point_status[i] = 1;
                        }
                    }
                }
            }
        });
#ifdef MPI_ON
        mpi.allReduce(v_point_cluster, magmaMPI::min);
        mpi.allReduce(v_point_status, magmaMPI::max);
#endif
        if (exa::reduce(v_point_status, 0, v_point_status.size(), 0) == 0) {
            is_done = true;
        } else {
            exa::for_each(0, v_point_id.size(), [&](int const &i) -> void {
                if (v_point_status[i] == 1) {
                    if (v_point_id[i] >= 0)
                        v_coord_cluster[v_point_id[i]] = v_point_cluster[i];
                    v_point_status[i] = 0;
                }
            });
        }
#ifdef MPI_ON
        if (iter_cnt == 0) is_done = false;
#endif
        ++iter_cnt;
    }
     */
    /*
#ifdef DEBUG_ON
    if (mpi.rank == 0)
        std::cout << "label iterations: " << iter_cnt << std::endl;
#endif
    int new_clusters = 0;
    exa::for_each(0, v_point_id.size(), [&](int const &i) -> void {
        if (v_point_nn[i] >= m && v_point_cluster[i] == i + cluster_size) {
            ++new_clusters;
        }
    });
#ifdef DEBUG_ON
    if (mpi.rank == 0)
        std::cout << "new clusters: " << new_clusters << std::endl;
#endif
    cluster_size += new_clusters;

    exa::for_each(0, v_point_id.size(), [&](int const &i) -> void {
        if (v_point_nn[i] >= m) {
    //            assert(v_point_cluster[i] != NO_CLUSTER);
            for (int j = 0; j < v_points_in_reach_size[i]; ++j) {
//                auto id2 = v_hit_table_id_2[v_points_in_reach_offset[i] + j];
                auto id2 = v_hit_table[v_points_in_reach_offset[i] + j];
                if (id2 == -1) continue;
                if (v_coord_cluster[id2] == NO_CLUSTER) {
                    v_coord_cluster[id2] = v_point_cluster[i];
                }
                else if (v_coord_cluster[id2] != v_point_cluster[i] && v_coord_nn[id2] >= m) {
    //                    std::cout << "CHECKPINT!!" << std::endl;
    //                    assert(v_point_cluster[i] < v_coord_cluster[id2]);
    //                    v_coord_cluster[id2] = v_point_cluster[i];
                }
            }
        }
    });
     */
}

void data_process::get_result_meta(int &cores, int &noise, int &clusters, int &n, magmaMPI mpi) noexcept {
    n = n_coord;
    auto const _m = m;
    cores = exa::count_if(v_coord_nn, 0, v_coord_nn.size(), [=]
#ifdef CUDA_ON
    __device__
#endif
    (int const &v) -> bool {
        return v >= _m;
    });

    noise = exa::count_if(v_coord_cluster, 0, v_coord_cluster.size(), [=]
#ifdef CUDA_ON
    __device__
#endif
    (int const &v) -> bool {
        return v >= 0;
    });

#ifdef MPI_ON
    d_vec<int> v_data(2);
    v_data[0] = cores;
    v_data[1] = noise;
    mpi.allReduce(v_data, magmaMPI::sum);
    cores = v_data[0];
    noise = v_data[1];
#endif

    /*
    d_vec<int> v_coord_cluster_cpy = v_coord_cluster;
    exa::sort(v_coord_cluster_cpy, 0, v_coord_cluster_cpy.size(), []
#ifdef CUDA_ON
    __device__
#endif
    (int const &v1, int const &v2) -> bool {
        return v1 < v2;
    });
    d_vec<int> v_iota(v_coord_cluster.size());
    exa::iota(v_iota, 0, v_iota.size(), 0);
    auto const it_coord_cluster_cpy = v_coord_cluster_cpy.begin();
    clusters = exa::count_if(v_iota, 1, v_iota.size(), [=]
#ifdef CUDA_ON
    __device__
#endif
    (int const &i) -> bool {
        if (it_coord_cluster_cpy[i] != it_coord_cluster_cpy[i-1])
            return true;
       return false;
    });
     */

/*
    std::unordered_map<int, int> v_cluster_map;
    for (int const &cluster : v_coord_cluster) {
//        if (cluster >= 0) {
            auto elem = v_cluster_map.find(cluster);
            if (elem == v_cluster_map.end()) {
                v_cluster_map.insert(std::make_pair(cluster, 1));
            } else {
                (*elem).second++;
            }
//        }
    }
    */
#ifdef MPI_ON
    d_vec<int> v_unique(v_iota.size());
    exa::unique(v_iota, v_unique, 0, v_iota.size(), 0, [&](auto const &i) -> bool {
        if (v_coord_cluster_cpy[i] != v_coord_cluster_cpy[i-1])
            return true;
        return false;
    });

    // TODO update for -1

//    std::cout << "clusters: " << clusters << " unique: " << v_unique.size() << " map: " << v_cluster_map.size() << std::endl;

    d_vec<int> v_node_cluster_size(mpi.n_nodes, 0);
    d_vec<int> v_node_cluster_offset(mpi.n_nodes, 0);
    v_node_cluster_size[mpi.rank] = clusters+1;
    mpi.allReduce(v_node_cluster_size, magmaMPI::sum);
    exa::exclusive_scan(v_node_cluster_size, v_node_cluster_offset, 0, v_node_cluster_size.size(), 0, 0);
    d_vec<int> v_node_unique(exa::reduce(v_node_cluster_size, 0, v_node_cluster_size.size(), 0));
#ifdef DEBUG_ON
    if (mpi.rank == 0)
        std::cout << "v_node_unique size: " << v_node_unique.size() << std::endl;
#endif
    std::copy(v_unique.begin(), v_unique.end(), std::next(v_node_unique.begin(), v_node_cluster_offset[mpi.rank]));
    mpi.allGatherv(v_node_unique, v_node_cluster_size, v_node_cluster_offset);
    exa::sort(v_node_unique, 0, v_node_unique.size(), [](int const &v1, int const &v2) -> bool {
        return v1 < v2;
    });
//    if (mpi.rank == 0)
//        magma_util::print_v(" clusters: ", &v_node_unique[0], v_node_unique.size());
    v_iota.resize(v_node_unique.size());
    exa::iota(v_iota, 0, v_iota.size(), 0);
    clusters = exa::count_if(v_iota, 1, v_iota.size(), [&](int const &i) -> bool {
        if (v_node_unique[i] != v_node_unique[i-1])
            return true;
        return false;
    });
#endif
}

void data_process::select_and_process(magmaMPI mpi) noexcept {
    v_coord_nn.resize(n_coord, 0);
    v_coord_cluster.resize(n_coord, NO_CLUSTER);
    v_coord_status.resize(n_coord, NOT_PROCESSED);

    d_vec<int> v_point_id(v_coord_id.size());
#ifdef DEBUG_ON
    if (mpi.rank == 0)
        std::cout << "number of cells: " << static_cast<int>(v_coord_cell_size.size()) << std::endl;
#endif
    exa::iota(v_point_id, 0, v_point_id.size(), 0);

    /*
//    int n_sample_size = n_coord * mpi.n_nodes / 100;
    // TODO Estimate this number better
    int n_sample_size = 3500000 * mpi.n_nodes;
#ifdef DEBUG_ON
    if (mpi.rank == 0)
        std::cout << "n_sample_size: " << n_sample_size << std::endl;
#endif
    d_vec<int> v_id_chunk(n_sample_size, -1);
    d_vec<float> v_data_chunk(n_sample_size * n_dim);
    int node_transmit_size = magma_util::get_block_size(mpi.rank, n_sample_size, mpi.n_nodes);
    int node_transmit_offset = magma_util::get_block_offset(mpi.rank, n_sample_size, mpi.n_nodes);

    int n_iter = 0;
//    while (n_iter < n_iter * node_transmit_size < n_coord / 2) {
    while (n_iter * mpi.n_nodes < 32) {
        if (mpi.rank == 0)
            std::cout << "n_iter: " << n_iter << std::endl;

        exa::fill(v_id_chunk, 0, v_id_chunk.size(), -1);
        exa::fill(v_data_chunk, 0, v_data_chunk.size(), static_cast<float>(-1));
        std::copy(std::next(v_point_id.begin(), n_iter * node_transmit_size),
                std::next(v_point_id.begin(), (n_iter+1) * node_transmit_size),
                std::next(v_id_chunk.begin(), node_transmit_offset));
        std::copy(std::next(v_coord.begin(), n_iter * node_transmit_size * n_dim),
                std::next(v_coord.begin(), (n_iter + 1) * node_transmit_size * n_dim),
                std::next(v_data_chunk.begin(), node_transmit_offset * n_dim));
#ifdef MPI_ON
        mpi.allGather(v_data_chunk);
#endif
        std::cout << "CHECKPOINT!" << std::endl;
        process_points(v_id_chunk, v_data_chunk, mpi);
        ++n_iter;
    }
    if (mpi.rank == 0)
        std::cout << "total iterations: "<< n_iter << std::endl;
    */

    d_vec<int> v_id_chunk;
    d_vec<float> v_data_chunk;
    int n_blocks = 1;
    for (int i = 0; i < n_blocks; ++i) {
        int block_size = magma_util::get_block_size(i, static_cast<int>(v_point_id.size()), n_blocks);
        int block_offset = magma_util::get_block_offset(i, static_cast<int>(v_point_id.size()), n_blocks);
        std::cout << "block offset: " << block_offset << " size: " << block_size << std::endl;
        v_id_chunk.clear();
        v_id_chunk.insert(v_id_chunk.begin(), std::next(v_point_id.begin(), block_offset),
                std::next(v_point_id.begin(), block_offset+block_size));
        v_data_chunk.clear();
        v_data_chunk.insert(v_data_chunk.begin(), std::next(v_coord.begin(), block_offset*n_dim),
                std::next(v_coord.begin(), (block_offset+block_size)*n_dim));
        process_points(v_id_chunk, v_data_chunk, mpi);
    }

}

void data_process::index_points(d_vec<float> &v_data, d_vec<int> &v_index) noexcept {
    /*
    s_vec<int> v_id(v_index.size());
    exa::iota(v_id, 0, v_id.size(), 0);
    exa::transform(v_id, v_index, 0, v_index.size(), 0,
            [&](int const &id) -> int {
                return cell_index(v_data[id * n_dim + v_dim_order[0]], v_min_bounds[v_dim_order[0]], e)
                       + (cell_index(v_data[id * n_dim + v_dim_order[1]], v_min_bounds[v_dim_order[1]], e) * v_dim_part_size[0]);
            });
            */
    auto const it_coords = v_data.begin();
    auto const it_index = v_index.begin();
    auto const _dim_0 = v_dim_order[0];
    auto const _dim_1 = v_dim_order[1];
    float const _bound_0 = v_min_bounds[_dim_0];
    float const _bound_1 = v_min_bounds[_dim_1];
    int const _mult = v_dim_part_size[0];
    int const _n_dim = n_dim;
    float const _e = e;
    exa::for_each(0, v_index.size(), [=]
#ifdef CUDA_ON
        __device__
#endif
    (int const &i) -> void {
        it_index[i] = (int)((it_coords[i * _n_dim + _dim_0] - _bound_0) / _e)
                + (int) ((it_coords[i * _n_dim + _dim_1] - _bound_1) / _e) * _mult;
//            *(it_index + i) = (int)( ( *(it_coords + (i * _n_dim + _dim_0)) - _bound_0 ) / _e )
//            + (int)( ( *(it_coords + (i * _n_dim + _dim_1)) - _bound_1 ) / _e ) * _mult;
    });

    /*
    thrust::transform(it_cnt_begin, it_cnt_end, v_index.begin(), [=]__device__(int const &i) -> int {
            return (int)( ( *(it_coords + (i * dim + dim_0)) - bound_0 ) / ee )
            + (int)( ( *(it_coords + (i * dim + dim_1)) - bound_1 ) / ee ) * mult;
    });
     */
}

void data_process::initialize_cells() noexcept {
    v_dim_part_size.resize(2);
    v_dim_part_size[0] = (v_max_bounds[v_dim_order[0]] - v_min_bounds[v_dim_order[0]]) / e + 1;
    v_dim_part_size[1] = (v_max_bounds[v_dim_order[1]] - v_min_bounds[v_dim_order[1]]) / e + 1;
    if (static_cast<uint64_t>(v_dim_part_size[0]) * v_dim_part_size[1] > INT32_MAX) {
        std::cerr << "FAIL: The epsilon value is too low and therefore not supported by the current version for the"
                     " input dataset (" << static_cast<uint64_t>(v_dim_part_size[0]) * v_dim_part_size[1] << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
    auto v_iota = v_coord_id;
    d_vec<int> v_point_cell_index(v_coord_id.size());

    index_points(v_coord, v_point_cell_index);

    auto const it_index = v_point_cell_index.begin();
    exa::sort(v_coord_id, 0, v_coord_id.size(), [=]
#ifdef CUDA_ON
        __device__
#endif
    (auto const &i1, auto const &i2) -> bool {
        return it_index[i1] < it_index[i2];
    });
    v_coord_cell_offset.resize(v_iota.size());
    v_coord_cell_offset[0] = 0;
    auto const it_coord_id = v_coord_id.begin();
    exa::copy_if(v_iota, 1, v_iota.size(), v_coord_cell_offset, 1,[=]
#ifdef CUDA_ON
        __device__
#endif
    (int const &i) -> bool {
        return it_index[it_coord_id[i]] != it_index[it_coord_id[i - 1]];
//        return *(it_index + *(it_coord_id + i)) != *(it_index + *(it_coord_id + i - 1));
    });
    v_coord_cell_size.resize(v_coord_cell_offset.size());
    auto const it_coord_cell_size = v_coord_cell_size.begin();
    auto const it_coord_cell_offset = v_coord_cell_offset.begin();
    exa::for_each(0, v_coord_cell_size.size() - 1, [=]
#ifdef CUDA_ON
        __device__
#endif
    (int const &i) -> void {
        it_coord_cell_size[i] = it_coord_cell_offset[i + 1] - it_coord_cell_offset[i];
//        *(it_coord_cell_size + i) = *(it_coord_cell_offset + i + 1) - *(it_coord_cell_offset + i);
    });
    v_coord_cell_size[v_coord_cell_size.size()-1] = n_coord - v_coord_cell_offset[v_coord_cell_size.size()-1];
    v_coord_cell_index.resize(v_coord_cell_offset.size());
    auto const it_coord_cell_index = v_coord_cell_index.begin();
    auto const it_point_cell_index = v_point_cell_index.begin();
    exa::for_each(0, v_coord_cell_index.size(), [=]
#ifdef CUDA_ON
        __device__
#endif
    (int const &i) -> void {
        it_coord_cell_index[i] = it_point_cell_index[it_coord_id[it_coord_cell_offset[i]]];
//        *(it_coord_cell_index + i) = *(it_point_cell_index + *(it_coord_id + *(it_coord_cell_offset + i)));
    });


    /*
    auto v_point_cell_index = v_coord_id;
    auto v_iota = v_coord_id;
    index_points(v_coord, v_point_cell_index);
    exa::sort(v_coord_id, 0, v_coord_id.size(), [&](auto const &i1, auto const &i2) -> bool {
        return v_point_cell_index[i1] < v_point_cell_index[i2];
    });

    v_coord_cell_offset.resize(v_iota.size());
    exa::unique(v_iota, v_coord_cell_offset, 0, v_iota.size(), 0, [&](auto const &i) -> bool {
        if (v_point_cell_index[v_coord_id[i]] != v_point_cell_index[v_coord_id[i-1]])
            return true;
        return false;
    });
    v_coord_cell_size.resize(v_coord_cell_offset.size());
    exa::iota(v_coord_cell_size, 0, v_coord_cell_size.size(), 0);
    exa::transform(v_coord_cell_size, v_coord_cell_size, 0, v_coord_cell_size.size()-1, 0,
            [&](auto const &i) -> int {
                return  v_coord_cell_offset[i+1] - v_coord_cell_offset[i];
            });
    v_coord_cell_size[v_coord_cell_size.size()-1] = n_coord - v_coord_cell_offset[v_coord_cell_size.size()-1];
    v_iota.resize(v_coord_cell_offset.size());
    v_coord_cell_index = v_iota;
    exa::transform(v_coord_cell_index, v_coord_cell_index, 0, v_coord_cell_index.size(), 0, [&](int const &i) -> int {
        return v_point_cell_index[v_coord_id[v_coord_cell_offset[i]]];
    });
     */
}
