//
// Created by Ernir Erlingsson on 19.8.2020.
//

#include <iostream>
#include <unordered_map>
#include "magma_util.h"
#include "nc_tree.h"
#ifdef OMP_ON
#include "magma_exa_omp.h"
#else
#include "magma_exa.h"
#endif


void nc_tree::determine_data_bounds() noexcept {
    v_min_bounds.resize(n_dim);
    v_max_bounds.resize(n_dim);
    v_coord_id.resize(n_coord);
    exa::iota(v_coord_id, 0, v_coord_id.size(), 0);
    exa::iota(v_min_bounds, 0, v_min_bounds.size(), 0);
    for (int d = 0; d < n_dim; ++d) {
        auto minmax = exa::minmax_element(v_coord_id, 0, v_coord_id.size(),
                [&](auto const i1, auto const i2) -> bool {
                    return v_coord[i1 * n_dim + d] < v_coord[i2 * n_dim + d];
                });
        v_min_bounds[d] = v_coord[(minmax.first * n_dim) + d];
        v_max_bounds[d] = v_coord[(minmax.second * n_dim) + d];
    }
    v_dim_order.resize(n_dim);
    exa::iota(v_dim_order, 0, v_dim_order.size(), 0);
    exa::sort(v_dim_order, 0, v_dim_order.size(), [&](int const &d1, int const &d2) -> bool {
       return (v_max_bounds[d1] - v_min_bounds[d1]) > (v_max_bounds[d2] - v_min_bounds[d2]);
    });
}

void nc_tree::collect_cells_in_reach(d_vec<int> &v_point_index, d_vec<int> &v_cell_reach,
        d_vec<int> &v_point_reach_offset, d_vec<int> &v_point_reach_size) noexcept {
    int const n_points = v_point_index.size();
    s_vec<int> v_point_reach_full(9 * n_points, -1);

    exa::for_each(0, n_points, [&](int const &i) -> void {
        if (v_point_index[i] < 0) {
            v_point_reach_size[i] = 0;
            return;
        }
        auto begin = std::next(v_point_reach_full.begin(), i * 9);
        auto i_index = begin;
        auto low = std::lower_bound(v_coord_cell_index.begin(), v_coord_cell_index.end(), v_point_index[i]-1);
        if (low != v_coord_cell_index.end() && *low == v_point_index[i]-1) {
            *(i_index++) = low - v_coord_cell_index.begin();
            ++low;
        }
        if (low != v_coord_cell_index.end() && *low == v_point_index[i]) {
            *(i_index++) = low - v_coord_cell_index.begin();
            ++low;
        }
        if (low != v_coord_cell_index.end() && *low == v_point_index[i]+1) {
            *(i_index++) = low - v_coord_cell_index.begin();
            ++low;
        }
        // above
        if (v_point_index[i] >= v_dim_part_size[0]) {
            low = std::lower_bound(v_coord_cell_index.begin(), v_coord_cell_index.end(), v_point_index[i]-v_dim_part_size[0]-1);
            if (low != v_coord_cell_index.end() && *low == v_point_index[i]-v_dim_part_size[0]-1) {
                *(i_index++) = low - v_coord_cell_index.begin();
                ++low;
            }
            if (low != v_coord_cell_index.end() && *low == v_point_index[i]-v_dim_part_size[0]) {
                *(i_index++) = low - v_coord_cell_index.begin();
                ++low;
            }
            if (low != v_coord_cell_index.end() && *low == v_point_index[i]-v_dim_part_size[0]+1) {
                *(i_index++) = low - v_coord_cell_index.begin();
                ++low;
            }
        }
        // below
        if (v_point_index[i] / v_dim_part_size[0] < v_dim_part_size[1]-1) {
            low = std::lower_bound(v_coord_cell_index.begin(), v_coord_cell_index.end(), v_point_index[i]+v_dim_part_size[0]-1);
            if (low != v_coord_cell_index.end() && *low == v_point_index[i]+v_dim_part_size[0]-1) {
                *(i_index++) = low - v_coord_cell_index.begin();
                ++low;
            }
            if (low != v_coord_cell_index.end() && *low == v_point_index[i]+v_dim_part_size[0]) {
                *(i_index++) = low - v_coord_cell_index.begin();
                ++low;
            }
            if (low != v_coord_cell_index.end() && *low == v_point_index[i]+v_dim_part_size[0]+1) {
                *(i_index++) = low - v_coord_cell_index.begin();
                ++low;
            }
        }
        v_point_reach_size[i] = i_index - begin;
    });

//    auto sum = exa::reduce(v_point_reach_size, 0, v_point_reach_size.size(), 0);
    exa::exclusive_scan(v_point_reach_size, v_point_reach_offset, 0, v_point_reach_size.size(), 0, 0);
    v_cell_reach.resize(v_point_reach_full.size());
    exa::copy_if(v_point_reach_full, v_cell_reach, 0, v_point_reach_full.size(), 0, [](int const &val) -> bool {
        return val >= 0;
    });
}

void nc_tree::process_points(d_vec<int> &v_point_id, d_vec<float> &v_point_data, magmaMPI mpi) noexcept {
    exa::for_each(0, v_point_id.size(), [&](int const &i) -> void {
        if (v_point_id[i] >= 0) {
            // local, we can modify the status
            v_coord_status[v_point_id[i]] = PROCESSED;
        }
    });
    // calculate cell index
    d_vec<int> v_point_index(v_point_id.size());
//    magma_util::measure_duration("Point Index: ", mpi.rank == 0, [&]() -> void {
        index_points(v_point_data, v_point_index);
//    });

    // obtain reach
    s_vec<int> v_point_cells_in_reach(v_point_id.size());
    s_vec<int> v_point_cell_reach_offset(v_point_id.size());
    s_vec<int> v_point_cell_reach_size(v_point_id.size());

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

    // calculate points in reach for each processed point
    exa::for_each(0, v_point_id.size(), [&](int const &i) -> void {
        auto p_sum = 0;
        for (int j = 0; j < v_point_cell_reach_size[i]; ++j) {
            p_sum += v_coord_cell_size[v_point_cells_in_reach[v_point_cell_reach_offset[i] + j]];
        }
        v_points_in_reach_size[i] = p_sum;
    });

//    exa::exclusive_scan(v_points_in_reach_size, v_points_in_reach_offset, 0, v_points_in_reach_size.size(), 0, 0);
//    long long table_size = exa::reduce(v_points_in_reach_size, 0, v_points_in_reach_size.size(), 0);
#ifdef DEBUG_ON
    if (mpi.rank == 0)
        std::cout << "table_size: " << table_size << std::endl;
#endif
//    d_vec<int> v_hit_table(table_size, -1);

    d_vec<int> v_point_nn(v_point_id.size(), 0);

    magma_util::measure_duration("Nearest Neighbour: ", mpi.rank == 0, [&]() -> void {
        exa::for_each_dynamic(0, v_point_id.size(), [&](int const &i) -> void {
//        if (v_points_in_reach_size[i] < m)
//            return;
            int nn = 0;
//        int cnt = 0;
            for (int ci = 0; ci < v_point_cell_reach_size[i]; ++ci) {
                int c = v_point_cells_in_reach[v_point_cell_reach_offset[i] + ci];
                for (int j = 0; j < v_coord_cell_size[c]; ++j) {
                    int id2 = v_coord_id[v_coord_cell_offset[c] + j];
                    if (dist_leq(&v_point_data[i * n_dim], &v_coord[id2 * n_dim], n_dim, e2)) {
                        ++nn;
//                    v_hit_table[v_points_in_reach_offset[i] + cnt] = id2;
                    }
                }
            }
            // used local var to minimize false sharing
            v_point_nn[i] = nn;
            if (v_point_nn[i] >= m && v_point_id[i] >= 0) {
                v_coord_nn[v_point_id[i]] = v_point_nn[i];
            }
        });
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

void nc_tree::get_result_meta(int &cores, int &noise, int &clusters, int &n, magmaMPI mpi) noexcept {
    n = n_coord;
    cores = 0;
    for (auto const &nn : v_coord_nn) {
        if (nn >= m) ++cores;
    }

    int cluster_points = 0;
    for (auto const &cluster : v_coord_cluster) {
        if (cluster >= 0) ++cluster_points;
    }
    noise = n_coord - cluster_points;

#ifdef MPI_ON
    d_vec<int> v_data(2);
    v_data[0] = cores;
    v_data[1] = noise;
    mpi.allReduce(v_data, magmaMPI::sum);
    cores = v_data[0];
    noise = v_data[1];
#endif

    d_vec<int> v_coord_cluster_cpy = v_coord_cluster;
    exa::sort(v_coord_cluster_cpy, 0, v_coord_cluster_cpy.size(), [](int const &v1, int const &v2) -> bool {
        return v1 < v2;
    });
    d_vec<int> v_iota(v_coord_cluster.size());
    exa::iota(v_iota, 0, v_iota.size(), 0);
    clusters = exa::count_if(v_iota, 1, v_iota.size(), [&](int const &i) -> bool {
        if (v_coord_cluster_cpy[i] != v_coord_cluster_cpy[i-1])
            return true;
       return false;
    });

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

void nc_tree::select_and_process(magmaMPI mpi) noexcept {
    v_coord_nn.resize(n_coord, 0);
    v_coord_cluster.resize(n_coord, NO_CLUSTER);
    v_coord_status.resize(n_coord, NOT_PROCESSED);


    d_vec<int> v_point_id(v_coord_id.size());
    if (mpi.rank == 0)
        std::cout << "number of cells: " << static_cast<int>(v_coord_cell_size.size()) << std::endl;
    exa::iota(v_point_id, 0, v_point_id.size(), 0);

//    int n_sample_size = static_cast<int>(v_coord_cell_size.size());
//    int n_sample_size = 131072 * mpi.n_nodes;
    int n_sample_size = n_coord * mpi.n_nodes / 100;
//    int n_sample_size = (n_coord / 4);
//#ifdef DEBUG_ON
    if (mpi.rank == 0)
        std::cout << "n_sample_size: " << n_sample_size << std::endl;
//#endif
    d_vec<int> v_id_chunk(n_sample_size, -1);
    d_vec<float> v_data_chunk(n_sample_size * n_dim);
    int node_transmit_size = magma_util::get_block_size(mpi.rank, n_sample_size, mpi.n_nodes);
    int node_transmit_offset = magma_util::get_block_offset(mpi.rank, n_sample_size, mpi.n_nodes);

    int n_iter = 0;
//    while (n_iter < n_iter * node_transmit_size < n_coord / 2) {
    while (n_iter < 10) {
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
        process_points(v_id_chunk, v_data_chunk, mpi);
        ++n_iter;
    }
    if (mpi.rank == 0)
        std::cout << "total iterations: "<< n_iter << std::endl;

    /*
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
     */
}

void nc_tree::index_points(d_vec<float> &v_data, d_vec<int> &v_index) noexcept {
    s_vec<int> v_id(v_index.size());
    exa::iota(v_id, 0, v_id.size(), 0);
    exa::transform(v_id, v_index, 0, v_index.size(), 0,
            [&](int const &id) -> int {
                return cell_index(v_data[id * n_dim + v_dim_order[0]], v_min_bounds[v_dim_order[0]], e)
                       + (cell_index(v_data[id * n_dim + v_dim_order[1]], v_min_bounds[v_dim_order[1]], e) * v_dim_part_size[0]);
            });
}

void nc_tree::initialize_cells() noexcept {
    v_dim_part_size.resize(2);
    v_dim_part_size[0] = (v_max_bounds[v_dim_order[0]] - v_min_bounds[v_dim_order[0]]) / e + 1;
    v_dim_part_size[1] = (v_max_bounds[v_dim_order[1]] - v_min_bounds[v_dim_order[1]]) / e + 1;
    if (static_cast<uint64_t>(v_dim_part_size[0]) * v_dim_part_size[1] > INT32_MAX) {
        std::cerr << "FAIL: The epsilon value is too low and therefore not supported by the current version for the"
                     " input dataset (" << static_cast<uint64_t>(v_dim_part_size[0]) * v_dim_part_size[1] << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
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

}
