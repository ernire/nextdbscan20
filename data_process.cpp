//
// Created by Ernir Erlingsson on 19.8.2020.
//

#include <iostream>
#include <stack>
#include "magma_util.h"
#include "data_process.h"
#ifdef OMP_ON
#include "magma_exa_omp.h"
#elif CUDA_ON
#include "magma_exa_cu.h"
#else
#include "magma_exa.h"
#endif

#ifdef CUDA_ON
void print_cuda_memory_usage() {
    size_t free_byte;
    size_t total_byte;
    auto cuda_status = cudaMemGetInfo( &free_byte, &total_byte );

    if ( cudaSuccess != cuda_status ) {
        printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status) );
        exit(1);
    }
    double free_db = (double)free_byte ;
    double total_db = (double)total_byte ;
    double used_db = total_db - free_db ;
    printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",
            used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);
}
#endif

void data_process::determine_data_bounds() noexcept {
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

void data_process::get_result_meta(long long &processed, int &cores, int &noise, int &clusters, int &n, magmaMPI mpi) noexcept {
    n = n_coord;
    auto const _m = m;
    cores = exa::count_if(v_coord_nn, 0, v_coord_nn.size(), [=]
#ifdef CUDA_ON
    __device__
#endif
    (int const &v) -> bool {
        return v >= _m;
    });

    noise = exa::count_if(v_coord_cluster_index, 0, v_coord_cluster_index.size(), []
#ifdef CUDA_ON
    __device__
#endif
    (int const &v) -> bool {
        return v == NO_CLUSTER;
    });

    processed = exa::count_if(v_coord_status, 0, v_coord_status.size(), [=]
#ifdef CUDA_ON
            __device__
#endif
    (int const &v) -> bool {
        return v != NOT_PROCESSED;
    });

#ifdef MPI_ON
    d_vec<int> v_data(3);
    v_data[0] = cores;
    v_data[1] = noise;
    v_data[2] = processed;
    if (mpi.n_nodes > 1)
        mpi.allReduce(v_data, magmaMPI::sum);
    cores = v_data[0];
    noise = v_data[1];
    processed = v_data[2];
#endif

    auto const it_cluster_label = v_cluster_label.begin();
    d_vec<int> v_cluster_iota(v_cluster_label.size());
    exa::iota(v_cluster_iota, 0, v_cluster_iota.size(), 0);
    clusters = exa::count_if(v_cluster_iota, 0, v_cluster_iota.size(), [=]
#ifdef CUDA_ON
        __device__
#endif
    (int const &i) -> bool {
        return it_cluster_label[i] == i;
    });

    // TODO write to file
}

void data_process::index_points(d_vec<float> const &v_data, d_vec<long long> &v_index) noexcept {
    auto const it_coords = v_data.begin();
    auto const it_index = v_index.begin();
    auto const _dim_0 = v_dim_order[0];
    auto const _dim_1 = v_dim_order[1];
    auto const _bound_0 = v_min_bounds[_dim_0];
    auto const _bound_1 = v_min_bounds[_dim_1];
    auto const _mult = v_dim_part_size[0];
    auto const _n_dim = n_dim;
    auto const _e = e;
    exa::for_each(0, v_index.size(), [=]
#ifdef CUDA_ON
        __device__
#endif
    (auto const &i) -> void {
        it_index[i] = (long long)((it_coords[i * _n_dim + _dim_0] - _bound_0) / _e)
                + (long long) ((it_coords[i * _n_dim + _dim_1] - _bound_1) / _e) * _mult;
    });
}

void data_process::build_nc_tree() noexcept {
    auto const _n_dim = n_dim;
    auto const _e_i = e / sqrtf(2);
    auto const _log2 = logf(2);
    auto const it_coord = v_coord.begin();
    auto const it_min_bound = v_min_bounds.begin();
    auto const _e = e;
    auto const _n_coord = n_coord;
    // calculate height
    d_vec<int> v_dim_height(n_dim);
    h_vec<int> v_dim_height_host(n_dim);
    auto const it_min_bounds = v_min_bounds;
    auto const it_max_bounds = v_max_bounds;
    for (int d = 0; d < n_dim; ++d) {
        float const max_limit = it_max_bounds[d] - it_min_bounds[d];
        v_dim_height_host[d] = ceilf(logf(max_limit / _e_i) / _log2) + 1;
    }
    v_dim_height = v_dim_height_host;

    auto const nc_height = exa::minmax_element(v_dim_height, 0, v_dim_height.size(), []
#ifdef CUDA_ON
            __device__
#endif
    (auto const &v1, auto const &v2) -> bool {
       return v1 < v2;
    }).second;
#ifdef DEBUG_ON
    std::cout << "nc tree height: " << nc_height << std::endl;
#endif
    // index size and offset
    v_nc_lvl_size.resize(nc_height);
    v_nc_lvl_offset.resize(nc_height);
    auto const it_nc_lvl_offset = v_nc_lvl_offset.begin();
    d_vec<int> v_base_index;

    d_vec<int> v_iota(n_coord);
    exa::iota(v_iota, 0, v_iota.size(), 0);
    for (int l = 0; l < nc_height; ++l) {
        auto _e_l = static_cast<float>(_e_i * pow(2, l));
        v_nc_lvl_offset[l] = l == 0? 0 : v_nc_lvl_offset[l-1] + v_nc_lvl_size[l-1];
        std::size_t n_lvl_size = l == 0 ? n_coord : v_nc_lvl_size[l-1];
        std::size_t n_lvl_offset = v_nc_lvl_offset[l];
        auto n_index_end = v_coord_cell_index.size();
        v_coord_cell_index.resize(v_coord_cell_index.size() + n_lvl_size);
        auto const it_coord_cell_index = v_coord_cell_index.begin();
        exa::iota(v_coord_cell_index, n_index_end, n_index_end + n_lvl_size, 0);
        auto it_coord_cell_offset = v_coord_cell_offset.begin();

        v_base_index.resize(n_lvl_size);
        auto const it_base_index = v_base_index.begin();
        if (l > 0) {
            exa::for_each(0, n_lvl_size, [=]
#ifdef CUDA_ON
            __device__
#endif
            (auto const &i) -> void {
                int level_mod = 1;
                int p_index = it_coord_cell_index[n_index_end + i];
                while (l - level_mod >= 0) {
                    if (l - level_mod < 1) {
                        p_index = it_coord_cell_index[it_coord_cell_offset[it_nc_lvl_offset[l - level_mod] + p_index]];
                    } else {
                        p_index = it_coord_cell_index[_n_coord + it_nc_lvl_offset[l - level_mod - 1] + it_coord_cell_offset[it_nc_lvl_offset[l - level_mod] + p_index]];
                    }
                    ++level_mod;
                }
                it_base_index[i] = p_index;
            });
        } else {
            exa::iota(v_base_index, 0, v_base_index.size(), 0);
        }
        exa::sort(v_coord_cell_index, n_index_end, n_index_end + n_lvl_size, [=]
#ifdef CUDA_ON
        __device__
#endif
        (auto const &i, auto const &j) -> bool {
            for (int d = 0; d < _n_dim; ++d) {
                if ((int)((it_coord[it_base_index[i] * _n_dim + d] - it_min_bound[d]) / _e_l)
                    < (int)((it_coord[it_base_index[j] * _n_dim + d] - it_min_bound[d]) / _e_l)) {
                    return true;
                } else if ((int)((it_coord[it_base_index[i] * _n_dim + d] - it_min_bound[d]) / _e_l)
                    > (int)((it_coord[it_base_index[j] * _n_dim + d] - it_min_bound[d]) / _e_l)) {
                    return false;
                }
            }
            return false;
        });
        std::size_t n_offset_size = v_coord_cell_offset.size();
        v_coord_cell_offset.resize(n_offset_size + n_lvl_size);
        v_coord_cell_offset[n_offset_size] = 0;
        exa::copy_if(v_iota, 1, n_lvl_size, v_coord_cell_offset, n_offset_size + 1,[=]
#ifdef CUDA_ON
        __device__
#endif
        (auto const &i) -> bool {
            for (int d = 0; d < _n_dim; ++d) {
                if ((int)((it_coord[it_base_index[it_coord_cell_index[i + n_index_end]] * _n_dim + d] - it_min_bound[d]) / _e_l)
                    != (int)((it_coord[it_base_index[it_coord_cell_index[i + n_index_end - 1]] * _n_dim + d] - it_min_bound[d]) / _e_l))
                    return true;
            }
            return false;
        });
        v_coord_cell_size.resize(v_coord_cell_offset.size());
        it_coord_cell_offset = v_coord_cell_offset.begin();
        auto const it_coord_cell_size = v_coord_cell_size.begin();
        exa::for_each(n_offset_size, v_coord_cell_size.size() - 1, [=]
#ifdef CUDA_ON
            __device__
#endif
        (auto const &i) -> void {
            it_coord_cell_size[i] = it_coord_cell_offset[i + 1] - it_coord_cell_offset[i];
        });
        v_coord_cell_size[v_coord_cell_size.size()-1] = n_lvl_size - v_coord_cell_offset[v_coord_cell_offset.size()-1];
        v_nc_lvl_size[l] = v_coord_cell_offset.size() - n_offset_size;

    }

    float const _min_float = -std::numeric_limits<float>::max();
    float const _max_float = std::numeric_limits<float>::max();

    auto const it_coord_cell_index = v_coord_cell_index.begin();
    auto const it_coord_cell_offset = v_coord_cell_offset.begin();
    // Determine the AABBs
    v_cell_AABB.resize(v_coord_cell_offset.size() * n_dim * 2);
    auto const it_cell_AABB = v_cell_AABB.begin();
    exa::for_each(0, v_coord_cell_offset.size(), [=]
#ifdef CUDA_ON
    __device__
#endif
    (auto const &i) -> void {
        for (int d = 0; d < _n_dim; ++d) {
            it_cell_AABB[(i * _n_dim * 2) + d] = _max_float;
        }
        for (int d = 0; d < _n_dim; ++d) {
            it_cell_AABB[(i * _n_dim * 2) + _n_dim + d] = _min_float;
        }
    });

    auto const it_coord_cell_size = v_coord_cell_size.begin();
    // level 0
    exa::for_each(0, v_nc_lvl_size[0], [=]
#ifdef CUDA_ON
        __device__
#endif
    (auto const &i) -> void {
        for (int j = 0; j < it_coord_cell_size[i]; ++j) {
            auto const p = it_coord_cell_index[it_coord_cell_offset[i] + j];
            for (int d = 0; d < _n_dim; ++d) {
                if (it_coord[p * _n_dim + d] - _e < it_cell_AABB[(i * _n_dim * 2) + d]) {
                    it_cell_AABB[(i * _n_dim * 2) + d] = it_coord[p * _n_dim + d] - _e;
                }
            }
            for (int d = 0; d < _n_dim; ++d) {
                if (it_coord[p * _n_dim + d] + _e > it_cell_AABB[(i * _n_dim * 2) + _n_dim + d]) {
                    it_cell_AABB[(i * _n_dim * 2) + _n_dim + d] = it_coord[p * _n_dim + d] + _e;
                }
            }
        }
    });

    for (int lvl = 1; lvl < nc_height; ++lvl) {
        exa::for_each(0, v_nc_lvl_size[lvl], [=]
#ifdef CUDA_ON
        __device__
#endif
        (auto const &i) -> void {
            for (int j = 0; j < it_coord_cell_size[it_nc_lvl_offset[lvl] + i]; ++j) {
                auto const c = it_coord_cell_index[_n_coord + it_nc_lvl_offset[lvl-1] + it_coord_cell_offset[it_nc_lvl_offset[lvl] + i] + j];
                for (int d = 0; d < _n_dim; ++d) {
                    if (it_cell_AABB[((it_nc_lvl_offset[lvl-1] + c) * _n_dim * 2) + d] < it_cell_AABB[((it_nc_lvl_offset[lvl] + i) * _n_dim * 2) + d]) {
                        it_cell_AABB[((it_nc_lvl_offset[lvl] + i) * _n_dim * 2) + d] = it_cell_AABB[((it_nc_lvl_offset[lvl-1] + c) * _n_dim * 2) + d];
                    }
                    if (it_cell_AABB[((it_nc_lvl_offset[lvl-1] + c) * _n_dim * 2) + _n_dim + d] > it_cell_AABB[((it_nc_lvl_offset[lvl] + i) * _n_dim * 2) + _n_dim + d]) {
                        it_cell_AABB[((it_nc_lvl_offset[lvl] + i) * _n_dim * 2) + _n_dim + d] = it_cell_AABB[((it_nc_lvl_offset[lvl-1] + c) * _n_dim * 2) + _n_dim + d];
                    }
                }
            }
        });
    }
}

void data_process::select_and_process(magmaMPI mpi) noexcept {
    // 128 MB sample size
//    int n_sample_size = 128000000 / (4 * n_dim);
    // process up to a million each iteration points
    int n_sample_size = 10000000;
    d_vec<int> v_id_chunk(n_sample_size, -1);
    d_vec<float> v_data_chunk(n_sample_size * n_dim);
    int node_transmit_size = magma_util::get_block_size(mpi.rank, n_sample_size, mpi.n_nodes);
    int node_transmit_offset = magma_util::get_block_offset(mpi.rank, n_sample_size, mpi.n_nodes);
//    std::cout << "node: " << mpi.rank << " with n_coord: " << n_coord << std::endl;
#ifdef DEBUG_ON
    std::cout << "node: " << mpi.rank << " transmit offset: " << node_transmit_offset << " size: " << node_transmit_size << " : " << n_coord << std::endl;
#endif
    d_vec<int> v_point_id(v_coord_id);
    exa::iota(v_point_id, 0, v_point_id.size(), 0);
    v_coord_nn.resize(n_coord, 0);
    v_coord_status.resize(n_coord, NOT_PROCESSED);
    v_coord_cluster_index.resize(n_coord, NO_CLUSTER);
    d_vec<int> v_point_nn(n_sample_size, 0);
    int track_height = static_cast<int>((v_nc_lvl_size.size() - 1));
    d_vec<int> v_tracker(n_sample_size * track_height * 2);
    std::size_t transmit_cnt = 0;
    int n_iter = 0;

    while (transmit_cnt < n_coord) {
        exa::fill(v_id_chunk, 0, v_id_chunk.size(), -1);
        exa::fill(v_data_chunk, 0, v_data_chunk.size(), std::numeric_limits<float>::max());
        exa::fill(v_point_nn, 0, v_point_nn.size(), 0);
        if (transmit_cnt + node_transmit_size <= n_coord) {
            exa::copy(v_point_id, transmit_cnt, transmit_cnt + node_transmit_size,
                    v_id_chunk, node_transmit_offset);
            exa::copy(v_coord, transmit_cnt * n_dim,
                    (transmit_cnt + node_transmit_size) * n_dim,
                    v_data_chunk, node_transmit_offset * n_dim);
        } else {
            std::size_t size = n_coord - transmit_cnt;
            exa::copy(v_point_id, transmit_cnt, transmit_cnt + size,
                    v_id_chunk, node_transmit_offset);
            exa::copy(v_coord, transmit_cnt * n_dim,
                    (transmit_cnt + size) * n_dim,
                    v_data_chunk, node_transmit_offset * n_dim);
        }
        transmit_cnt += node_transmit_size;
#ifdef DEBUG_ON
//        if (mpi.rank == 0)
//            std::cout << "transmit iter: " << n_iter << ", elems sent:" << transmit_cnt << std::endl;
#endif
#ifdef MPI_ON
        if (mpi.n_nodes > 1)
            mpi.allGather(v_data_chunk);
#endif
        process_points(v_id_chunk, v_data_chunk, v_point_nn, v_tracker, track_height, mpi);
        ++n_iter;
    }
}

void data_process::process_points(d_vec<int> const &v_point_id, d_vec<float> const &v_point_data, d_vec<int> &v_point_nn,
        d_vec<int> &v_tracker, int const track_height, magmaMPI mpi) noexcept {
    auto const it_point_id = v_point_id.begin();
    auto const it_point_data = v_point_data.begin();
    auto const it_point_nn = v_point_nn.begin();
    auto const it_coord_nn = v_coord_nn.begin();
    auto const it_coord_cell_size = v_coord_cell_size.begin();
    auto const it_nc_lvl_size = v_nc_lvl_size.begin();
    auto const it_nc_lvl_offset = v_nc_lvl_offset.begin();
    auto const it_coord_cell_index = v_coord_cell_index.begin();
    auto const it_coord_status = v_coord_status.begin();
    auto const it_coord_cell_offset = v_coord_cell_offset.begin();
    auto const it_coord = v_coord.begin();
    auto const it_cell_AABB = v_cell_AABB.begin();
    auto const _n_dim = n_dim;
    auto const _n_coord = n_coord;
    auto const _m = m;
    auto const _e2 = e2;

    auto const it_tracker = v_tracker.begin();
    float _max_float = std::numeric_limits<float>::max();

    exa::for_each(0, v_point_id.size(), [=]
#ifdef CUDA_ON
    __device__
#endif
    (auto const &i) -> void {
        int hits = 0;
        auto const p = &it_point_data[i * _n_dim];
        if (p[0] == _max_float) {
            return;
        }
        auto const stack = &it_tracker[i * track_height * 2];
        for (int c1 = 0; c1 < it_nc_lvl_size[track_height - 1]; ++c1) {
            int lvl = track_height - 1;
            stack[lvl * 2] = c1;
            stack[lvl * 2 + 1] = 0;
            do {
                int c = stack[lvl * 2];
                int j = stack[lvl * 2 + 1];
                bool are_connected = true;
                if (j == 0) {
                    auto const min2 = &it_cell_AABB[((it_nc_lvl_offset[lvl] + c) * _n_dim * 2)];
                    auto const max2 = &it_cell_AABB[((it_nc_lvl_offset[lvl] + c) * _n_dim * 2) + _n_dim];
                    for (int d = 0; d < _n_dim; ++d) {
                        if (p[d] < min2[d] || p[d] > max2[d]) {
                            are_connected = false;
                            break;
                        }
                    }
                }
                if (are_connected) {
                    if (j < it_coord_cell_size[it_nc_lvl_offset[lvl] + c]) {
                        stack[lvl * 2 + 1] = j + 1;
                        if (lvl == 0) {
                            for (int k = 0; k < it_coord_cell_size[c]; ++k) {
                                auto const p2 = &it_coord[it_coord_cell_index[it_coord_cell_offset[c] + k] * _n_dim];
                                float length = 0;
                                for (int d = 0; d < _n_dim; ++d) {
                                    length += (p[d] - p2[d]) * (p[d] - p2[d]);
                                    if (length > _e2) {
                                        break;
                                    }
                                }
                                if (length <= _e2) {
                                    if (++hits == _m) {
                                        it_point_nn[i] = _m;
                                        return;
                                    }
                                }
                            }
                            ++lvl;
                        } else {
                            --lvl;
                            stack[lvl * 2] = it_coord_cell_index[_n_coord + it_nc_lvl_offset[lvl] + it_coord_cell_offset[it_nc_lvl_offset[lvl+1] + c] + j];
                            stack[lvl * 2 + 1] = 0;
                        }
                    } else {
                        ++lvl;
                    }
                } else {
                    ++lvl;
                }
            } while (lvl < track_height);
            it_point_nn[i] = hits;
        }
    });

#ifdef MPI_ON
    if (mpi.n_nodes > 1)
        mpi.allReduce(v_point_nn, magmaMPI::sum);
#endif

    exa::for_each(0, v_point_id.size(), [=]
#ifdef CUDA_ON
        __device__
#endif
    (auto const &i) -> void {
        if (it_point_id[i] != -1 && it_point_nn[i] >= _m) {
            it_coord_nn[it_point_id[i]] = _m;
        }
    });

    // Copy only the cores
    d_vec<int> v_point_iota(v_point_id.size());
    d_vec<int> v_point_core_id(v_point_id.size());
    auto const it_point_core_id = v_point_core_id.begin();
    exa::iota(v_point_iota, 0, v_point_iota.size(), 0);
    exa::copy_if(v_point_iota, 0, v_point_iota.size(), v_point_core_id, 0, [=]
#ifdef CUDA_ON
        __device__
#endif
    (auto const &i) -> bool {
        if (it_point_nn[i] >= _m) {
            return true;
        }
        return false;
    });

    d_vec<int> v_core_cluster_index(v_point_core_id.size(), NO_CLUSTER);
    auto const it_core_cluster_index = v_core_cluster_index.begin();
    auto const it_coord_cluster_index = v_coord_cluster_index.begin();
    d_vec<int> v_point_new_cluster_mark(v_point_core_id.size(), 0);
    d_vec<int> v_point_new_cluster_offset(v_point_core_id.size(), 0);
    auto const it_point_new_cluster_mark = v_point_new_cluster_mark.begin();

    exa::for_each(0, v_point_core_id.size(), [=]
#ifdef CUDA_ON
        __device__
#endif
    (auto const &k) -> void {
        auto i = it_point_core_id[k];
        if (it_point_id[i] >= 0) {
            // Discover which new cores have not been labeled
            if (it_coord_cluster_index[it_point_id[i]] == NO_CLUSTER) {
                // mark for a new label
                it_point_new_cluster_mark[k] = 1;
            }
            else {
                it_core_cluster_index[k] = it_coord_cluster_index[it_point_id[i]];
            }
        }
    });

#ifdef MPI_ON
    if (mpi.n_nodes > 1)
        mpi.allReduce(v_point_new_cluster_mark, magmaMPI::max);
#endif

    // count the new labels
    auto new_cluster_cores = exa::count_if(v_point_new_cluster_mark, 0, v_point_new_cluster_mark.size(), []
#ifdef CUDA_ON
            __device__
#endif
    (auto const &v) -> bool {
        return v == 1;
    });
#ifdef DEBUG_ON
    if (mpi.rank == 0)
        std::cout << "new cluster cores: " << new_cluster_cores << std::endl;
#endif

    int cluster_index_begin = static_cast<int>(v_cluster_label.size());
    v_cluster_label.resize(cluster_index_begin + new_cluster_cores);
    auto const it_cluster_label = v_cluster_label.begin();
    // Create new label ids
    exa::iota(v_cluster_label, cluster_index_begin, v_cluster_label.size(), cluster_index_begin);
    // the new label indexes
    exa::exclusive_scan(v_point_new_cluster_mark, 0, v_point_new_cluster_mark.size(), v_point_new_cluster_offset,
            0, cluster_index_begin);
    auto const it_point_new_cluster_offset = v_point_new_cluster_offset.begin();

    exa::for_each(0, v_point_core_id.size(), [=]
#ifdef CUDA_ON
        __device__
#endif
    (auto const &k) -> void {
        auto i = it_point_core_id[k];
        if (it_point_new_cluster_mark[k] == 1) {
            // mark the new cluster indexes
            it_core_cluster_index[k] = it_point_new_cluster_offset[k];
            if (it_point_id[i] >= 0) {
                it_coord_cluster_index[it_point_id[i]] = it_core_cluster_index[k];
            }
        }
    });


#ifdef MPI_ON
    if (mpi.n_nodes > 1)
            mpi.allReduce(v_core_cluster_index, magmaMPI::min);
#endif

    int iter_cnt = 0;
    d_vec<int> v_running(1);
    auto const it_running = v_running.begin();
    do {
#ifdef DEBUG_ON
        if (mpi.rank == 0)
            std::cout << "iter: " << ++iter_cnt << std::endl;
#endif
        // set lowest
        v_running[0] = 0;
        exa::for_each(0, v_point_core_id.size(), [=]
#ifdef CUDA_ON
        __device__
#endif
        (auto const &ii) -> void {
            auto const i = it_point_core_id[ii];
            auto const p = &it_point_data[i * _n_dim];
            auto const stack = &it_tracker[i * track_height * 2];
            for (int c1 = 0; c1 < it_nc_lvl_size[track_height - 1]; ++c1) {
                int lvl = track_height - 1;
                stack[lvl * 2] = c1;
                stack[lvl * 2 + 1] = 0;
                do {
                    int c = stack[lvl * 2];
                    int j = stack[lvl * 2 + 1];
                    bool are_connected = true;
                    if (j == 0) {
                        auto const min2 = &it_cell_AABB[((it_nc_lvl_offset[lvl] + c) * _n_dim * 2)];
                        auto const max2 = &it_cell_AABB[((it_nc_lvl_offset[lvl] + c) * _n_dim * 2) + _n_dim];
                        for (int d = 0; d < _n_dim; ++d) {
                            if (p[d] < min2[d] || p[d] > max2[d]) {
                                are_connected = false;
                                break;
                            }
                        }
                    }
                    if (are_connected) {
                        if (j < it_coord_cell_size[it_nc_lvl_offset[lvl] + c]) {
                            stack[lvl * 2 + 1] = j + 1;
                            if (lvl == 0) {
                                for (int k = 0; k < it_coord_cell_size[c]; ++k) {
                                    auto const id2 = it_coord_cell_index[it_coord_cell_offset[c] + k];
                                    if (it_coord_nn[id2] < _m)
                                        continue;

                                    if (it_cluster_label[it_coord_cluster_index[id2]] == it_cluster_label[it_core_cluster_index[ii]]) {
                                        continue;
                                    }
                                    if (it_cluster_label[it_coord_cluster_index[id2]] > it_cluster_label[it_core_cluster_index[ii]]
                                        && it_coord_status[id2] != PROCESSED) {
                                            continue;
                                    }
                                    auto const p2 = &it_coord[id2 * _n_dim];
                                    float length = 0;
                                    for (int d = 0; d < _n_dim; ++d) {
                                        length += (p[d] - p2[d]) * (p[d] - p2[d]);
                                        if (length > _e2) {
                                            break;
                                        }
                                    }
                                    if (length <= _e2) {
                                        if (it_cluster_label[it_coord_cluster_index[id2]] > it_cluster_label[it_core_cluster_index[ii]]) {
                                            exa::atomic_min(&it_cluster_label[it_coord_cluster_index[id2]],
                                                    it_cluster_label[it_core_cluster_index[ii]]);
                                            it_running[0] = 1;
                                        } else {
                                            exa::atomic_min(&it_cluster_label[it_core_cluster_index[ii]],
                                                    it_cluster_label[it_coord_cluster_index[id2]]);
                                            it_running[0] = 1;
                                        }
                                    }
                                }
                                ++lvl;
                            } else {
                                --lvl;
                                stack[lvl * 2] = it_coord_cell_index[
                                        _n_coord + it_nc_lvl_offset[lvl] +
                                        it_coord_cell_offset[it_nc_lvl_offset[lvl + 1] + c] + j
                                ];
                                stack[lvl * 2 + 1] = 0;
                            }
                        } else {
                            ++lvl;
                        }
                    } else {
                        ++lvl;
                    }
                } while (lvl < track_height);
            }
        });

#ifdef MPI_ON
        if (mpi.n_nodes > 1)
            mpi.allReduce(v_cluster_label, magmaMPI::min);
#endif

        // flatten
        exa::for_each(0, v_core_cluster_index.size(), [=]
#ifdef CUDA_ON
        __device__
#endif
        (auto const &k) -> void {
            while (it_cluster_label[it_core_cluster_index[k]] != it_core_cluster_index[k]) {
                it_core_cluster_index[k] = it_cluster_label[it_core_cluster_index[k]];
            }
        });

        exa::for_each(0, v_coord_id.size(), [=]
#ifdef CUDA_ON
        __device__
#endif
        (auto const &i) -> void {
            if (it_coord_cluster_index[i] == NO_CLUSTER)
                return;
            while (it_cluster_label[it_coord_cluster_index[i]] != it_coord_cluster_index[i]) {
                it_coord_cluster_index[i] = it_cluster_label[it_coord_cluster_index[i]];
            }
        });

#ifdef MPI_ON
        mpi.allReduce(v_running, magmaMPI::max);
#endif

    } while (v_running[0] == 1);

    exa::for_each(0, v_point_core_id.size(), [=]
#ifdef CUDA_ON
            __device__
#endif
    (auto const &k) -> void {
        auto i = it_point_core_id[k];
        if (it_point_id[i] >= 0) {
            it_coord_cluster_index[it_point_id[i]] = it_core_cluster_index[k];
        }
    });

    exa::for_each(0, v_point_id.size(), [=]
#ifdef CUDA_ON
    __device__
#endif
    (auto const &i) -> void {
        if (it_point_id[i] >= 0) {
            it_coord_status[it_point_id[i]] = PROCESSED;
        }
    });

    // Label non-cores
    exa::for_each(0, v_point_core_id.size(), [=]
#ifdef CUDA_ON
    __device__
#endif
    (auto const &ii) -> void {
        auto const i = it_point_core_id[ii];
        auto const p = &it_point_data[i * _n_dim];
        auto const stack = &it_tracker[i * track_height * 2];
        for (int c1 = 0; c1 < it_nc_lvl_size[track_height - 1]; ++c1) {
            int lvl = track_height - 1;
            stack[lvl * 2] = c1;
            stack[lvl * 2 + 1] = 0;
            do {
                int c = stack[lvl * 2];
                int j = stack[lvl * 2 + 1];
                bool are_connected = true;
                if (j == 0) {
                    auto const min2 = &it_cell_AABB[((it_nc_lvl_offset[lvl] + c) * _n_dim * 2)];
                    auto const max2 = &it_cell_AABB[((it_nc_lvl_offset[lvl] + c) * _n_dim * 2) + _n_dim];
                    for (int d = 0; d < _n_dim; ++d) {
                        if (p[d] < min2[d] || p[d] > max2[d]) {
                            are_connected = false;
                            break;
                        }
                    }
                }
                if (are_connected) {
                    if (j < it_coord_cell_size[it_nc_lvl_offset[lvl] + c]) {
                        stack[lvl * 2 + 1] = j + 1;
                        if (lvl == 0) {
                            for (int k = 0; k < it_coord_cell_size[c]; ++k) {
                                auto const id2 = it_coord_cell_index[it_coord_cell_offset[c] + k];
                                if (it_coord_cluster_index[id2] != NO_CLUSTER)
                                    continue;
                                auto const p2 = &it_coord[id2 * _n_dim];
                                float length = 0;
                                for (int d = 0; d < _n_dim; ++d) {
                                    length += (p[d] - p2[d]) * (p[d] - p2[d]);
                                    if (length > _e2) {
                                        break;
                                    }
                                }
                                if (length <= _e2) {
                                    exa::atomic_min(&it_coord_cluster_index[id2],
                                            it_cluster_label[it_core_cluster_index[ii]]);
                                }
                            }
                            ++lvl;
                        } else {
                            --lvl;
                            stack[lvl * 2] = it_coord_cell_index[
                                    _n_coord + it_nc_lvl_offset[lvl] + it_coord_cell_offset[it_nc_lvl_offset[lvl+1] + c] + j
                            ];
                            stack[lvl * 2 + 1] = 0;
                        }
                    } else {
                        ++lvl;
                    }
                } else {
                    ++lvl;
                }
            } while (lvl < track_height);
        }
    });
}

