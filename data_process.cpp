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

void data_process::collect_cells_in_reach(d_vec<long long> const &v_point_index, d_vec<int> &v_cell_reach,
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
    auto const _coord_cell_index_size = v_coord_cell_index.size();
    exa::for_each(0, v_point_index.size(), [=]
#ifdef CUDA_ON
        __device__
#endif
    (auto const &i) -> void {
        if (it_point_index[i] < 0) {
            it_point_reach_size[i] = 0;
            return;
        }
        auto it_begin = it_full_reach + (i * 9);
        auto it_out = it_begin;
        int val = *(it_bounds + (i * 3));
        if (val < _coord_cell_index_size && *(it_coord_index + val) == *(it_point_index + i) - 1) {
            *(it_out++) = val;
            ++val;
        }
        if (val < _coord_cell_index_size && *(it_coord_index + val) == *(it_point_index + i)) {
            *(it_out++) = val;
            ++val;
        }
        if (val < _coord_cell_index_size && *(it_coord_index + val) == *(it_point_index + i) + 1) {
            *(it_out++) = val;
            ++val;
        }
        val = *(it_bounds + (i * 3) + 1);
        if (val < _coord_cell_index_size && *(it_coord_index + val) == *(it_point_index + i) - _dim_part_0 - 1) {
            *(it_out++) = val;
            ++val;
        }
        if (val < _coord_cell_index_size && *(it_coord_index + val) == *(it_point_index + i) - _dim_part_0) {
            *(it_out++) = val;
            ++val;
        }
        if (val < _coord_cell_index_size && *(it_coord_index + val) == *(it_point_index + i) - _dim_part_0 + 1) {
            *(it_out++) = val;
            ++val;
        }
        val = *(it_bounds + (i * 3) + 2);
        if (val < _coord_cell_index_size && *(it_coord_index + val) == *(it_point_index + i) + _dim_part_0 - 1) {
            *(it_out++) = val;
            ++val;
        }
        if (val < _coord_cell_index_size && *(it_coord_index + val) == *(it_point_index + i) + _dim_part_0) {
            *(it_out++) = val;
            ++val;
        }
        if (val < _coord_cell_index_size && *(it_coord_index + val) == *(it_point_index + i) + _dim_part_0 + 1) {
            *(it_out++) = val;
        }
        *(it_point_reach_size + i) = it_out - it_begin;
    });

    exa::exclusive_scan(v_point_reach_size, 0, v_point_reach_size.size(), v_point_reach_offset, 0, 0);
    v_cell_reach.resize(v_point_reach_full.size());
    exa::copy_if(v_point_reach_full, 0, v_point_reach_full.size(), v_cell_reach, 0, []
#ifdef CUDA_ON
    __device__
#endif
    (auto const &val) -> bool {
        return val >= 0;
    });
}

void data_process::process_points3(d_vec<int> const &v_point_coord_id, d_vec<float> const &v_point_data,
        d_vec<long long> &v_point_index, d_vec<int> &v_point_cells_in_reach,
        d_vec<int> &v_point_cell_reach_offset, d_vec<int> &v_point_cell_reach_size,
        magmaMPI mpi) noexcept {

    // calculate cell index
    index_points(v_point_data, v_point_index);

    // obtain reach
    collect_cells_in_reach(v_point_index, v_point_cells_in_reach, v_point_cell_reach_offset,
            v_point_cell_reach_size);

    d_vec<int> v_point_nn(v_point_coord_id.size(), 0);
    auto const it_coord_id = v_coord_id.begin();
    auto const it_point_cell_reach_size = v_point_cell_reach_size.begin();
    auto const it_point_cells_in_reach = v_point_cells_in_reach.begin();
    auto const it_point_cell_reach_offset = v_point_cell_reach_offset.begin();
    auto const it_coord_cell_size = v_coord_cell_size.begin();
    auto const it_coord_cell_offset = v_coord_cell_offset.begin();
    auto const it_point_data = v_point_data.begin();
    auto const it_coord_data = v_coord.begin();
    auto const it_coord_status = v_coord_status.begin();
    auto const it_point_nn = v_point_nn.begin();
    auto const it_coord_nn = v_coord_nn.begin();
    auto const it_point_id = v_point_coord_id.begin();
    auto const _n_dim = n_dim;
    auto const _e2 = e2;
    auto const _e_inner = (e / sqrtf(2)) / 2;
    auto const _m = m;
    int const _point_size = static_cast<int>(v_point_coord_id.size());

    exa::for_each(0, v_point_coord_id.size(), [=]
#ifdef CUDA_ON
        __device__
#endif
    (auto const &i) -> void {
        if (it_point_id[i] >= 0) {
            it_coord_status[it_point_id[i]] = MARKED;
        }
    });

    d_vec<int> v_cell_point_id(v_point_cells_in_reach.size());
    auto const it_cell_point_id = v_cell_point_id.begin();
    exa::for_each(0, v_point_coord_id.size(), [=]
#ifdef CUDA_ON
            __device__
#endif
    (auto const &i) -> void {
        for (int j = 0; j < it_point_cell_reach_size[i]; ++j) {
            it_cell_point_id[it_point_cell_reach_offset[i] + j] = i;
        }
    });

    /*
    exa::for_each(0, v_point_cells_in_reach.size(), [=]
#ifdef CUDA_ON
        __device__
#endif
    (auto const &i) -> void {
        int p_i = it_cell_point_id[i];
        auto const c = it_point_cells_in_reach[i];
        for (int j = 0; j < it_coord_cell_size[c]; ++j) {
            int const id2 = it_coord_id[it_coord_cell_offset[c] + j];
            if (it_coord_status[id2] == PROCESSED)
                continue;
            if (it_coord_status[id2] == MARKED && it_point_id[p_i] >= 0 && it_point_id[p_i] >= id2)
                continue;
            float length = 0;
            for (int d = 0; d < _n_dim; ++d) {
                length += (it_point_data[p_i * _n_dim + d] - it_coord_data[id2 * _n_dim + d]) *
                          (it_point_data[p_i * _n_dim + d] - it_coord_data[id2 * _n_dim + d]);
                if (length > _e2)
                    break;
            }
            if (length <= _e2) {
                if (it_point_id[p_i] >= 0) {
                    exa:atomic_add(&it_coord_nn[it_point_id[p_i]], 1);
                }
                exa::atomic_add(&it_coord_nn[id2], 1);
            }
        }
    });
     */

    exa::for_each(0, v_point_coord_id.size(), [=]
#ifdef CUDA_ON
        __device__
#endif
    (auto const &i) -> void {
        if (it_point_id[i] >= 0) {
            it_point_nn[i] = it_coord_nn[it_point_id[i]];
        }
    });

    // TODO MPI collect hits

    d_vec<int> v_point_iota(v_point_coord_id.size());
    d_vec<int> v_point_core_id(v_point_coord_id.size());
    auto const it_point_core_id = v_point_core_id.begin();
    exa::iota(v_point_iota, 0, v_point_iota.size(), 0);
    // copy only the cores for further processing
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
    // setja nýtt core cluster label ef þörf er á
    // TODO move these allocations to parent
    d_vec<int> v_point_cluster_index(v_point_core_id.size(), INT32_MAX);
    auto const it_point_cluster_index = v_point_cluster_index.begin();
    auto const it_coord_cluster_index = v_coord_cluster_index.begin();

    // enumerate labels
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
            } else {
                it_point_cluster_index[k] = it_coord_cluster_index[it_point_id[i]];
            }
        }
    });

    // TODO MPI MERGE

    // count the new labels
    auto new_cluster_cores = exa::count_if(v_point_new_cluster_mark, 0, v_point_new_cluster_mark.size(), []
#ifdef CUDA_ON
            __device__
#endif
    (auto const &v) -> bool {
        return v == 1;
    });
#ifdef DEBUG_ON
//    std::cout << "new cluster cores: " << new_cluster_cores << std::endl;
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
    auto const _cluster_label_size = v_cluster_label.size();
    exa::for_each(0, v_point_core_id.size(), [=]
#ifdef CUDA_ON
            __device__
#endif
    (auto const &k) -> void {
        auto i = it_point_core_id[k];
        if (it_point_new_cluster_mark[k] == 1) {
            // mark the new cluster indexes
            it_point_cluster_index[k] = it_point_new_cluster_offset[k];
            if (it_point_id[i] >= 0) {
                it_coord_cluster_index[it_point_id[i]] = it_point_cluster_index[k];
            }
        }
    });

    // scan for cluster mergers

    // collect all the cluster mergers

    // remove duplicates

    // merge


    /*
    exa::for_each(0, v_point_core_id.size(), [=]
#ifdef CUDA_ON
        __device__
#endif
    (auto const &k) -> void {
        auto i = it_point_core_id[k];
        for (int ci = 0; ci < it_point_cell_reach_size[i]; ++ci) {
            int const c = it_point_cells_in_reach[it_point_cell_reach_offset[i] + ci];
            for (int j = 0; j < it_coord_cell_size[c]; ++j) {
                int const id2 = it_coord_id[it_coord_cell_offset[c] + j];
                if (it_coord_nn[id2] >= _m) continue;
                if (it_coord_cluster_index[id2] == NO_CLUSTER) {
                    float length = 0;
                    for (int d = 0; d < _n_dim; ++d) {
                        length += (it_point_data[i * _n_dim + d] - it_coord_data[id2 * _n_dim + d]) *
                                  (it_point_data[i * _n_dim + d] - it_coord_data[id2 * _n_dim + d]);
                        if (length > _e2)
                            break;
                    }
                    if (length <= _e2) {
                        exa::atomic_min(&it_coord_cluster_index[id2],
                                it_cluster_label[it_point_cluster_index[k]]);
                    }
                }
            }
        }
    });
     */

    exa::for_each(0, v_point_coord_id.size(), [=]
#ifdef CUDA_ON
        __device__
#endif
        (auto const &i) -> void {
        if (it_point_id[i] >= 0) {
            it_coord_status[it_point_id[i]] = PROCESSED;
        }
    });

}

void data_process::process_points2(d_vec<int> const &v_point_coord_id, d_vec<float> const &v_point_data,
        d_vec<long long> &v_point_index, d_vec<int> &v_point_cells_in_reach,
        d_vec<int> &v_point_cell_reach_offset, d_vec<int> &v_point_cell_reach_size,
        magmaMPI mpi) noexcept {

    // calculate cell index
    index_points(v_point_data, v_point_index);
    /*
    auto const it_point_index = v_point_index.begin();

    d_vec<int> v_point_index_iota(v_point_id.size());
    exa::iota(v_point_index_iota, 0, v_point_index_iota.size(), 0);
    exa::sort(v_point_index_iota, 0, v_point_index_iota.size(), [=]
#ifdef CUDA_ON
        __device__
#endif
    (auto const &i1, auto const &i2) -> bool {
        return it_point_index[i1] < it_point_index[i2];
    });
//    v_coord_cell_offset.resize(v_iota.size());
//    v_coord_cell_offset[0] = 0;
    d_vec<int> v_point_index_offset(v_point_index_iota.size());
    v_point_index_offset[0] = 0;
    exa::copy_if(v_point_index_iota, 1, v_point_index_iota.size(), v_coord_cell_offset, 1,[=]
#ifdef CUDA_ON
        __device__
#endif
    (auto const &i) -> bool {
        return it_point_index[i] != it_point_index[i-1];
    });
    d_vec<int> v_point_index_size(v_point_index_offset.size());
    auto const it_point_index_size = v_point_index_size.begin();
    auto const it_point_index_offset = v_point_index_offset.begin();
    exa::for_each(0, v_coord_cell_size.size() - 1, [=]
#ifdef CUDA_ON
            __device__
#endif
    (auto const &i) -> void {
        it_point_index_size[i] = it_point_index_offset[i + 1] - it_point_index_offset[i];
    });
    v_point_index_size[v_point_index_size.size()-1] = v_point_index.size() - v_point_index_offset[v_point_index_size.size()-1];
    d_vec<int> v_point_index_pruned(v_point_index.size());
//    exa::unique(v_point_index, 0, v_point_index.size(), v_point_index_pruned, 0, )
    */
    // obtain reach
    /*
    collect_cells_in_reach(v_point_index, v_point_cells_in_reach, v_point_cell_reach_offset,
            v_point_cell_reach_size);

    d_vec<int> v_point_nn(v_point_coord_id.size(), 0);
    auto const it_coord_id = v_coord_id.begin();
    auto const it_point_cell_reach_size = v_point_cell_reach_size.begin();
    auto const it_point_cells_in_reach = v_point_cells_in_reach.begin();
    auto const it_point_cell_reach_offset = v_point_cell_reach_offset.begin();
    auto const it_coord_cell_size = v_coord_cell_size.begin();
    auto const it_coord_cell_offset = v_coord_cell_offset.begin();
    auto const it_point_data = v_point_data.begin();
    auto const it_coord_data = v_coord.begin();
    auto const it_point_nn = v_point_nn.begin();
    auto const it_coord_nn = v_coord_nn.begin();
    auto const it_point_id = v_point_coord_id.begin();
    auto const _n_dim = n_dim;
    auto const _e2 = e2;
    auto const _e_inner = (e / sqrtf(2)) / 2;
    auto const _m = m;

    d_vec<int> v_max_point_reach(v_point_coord_id.size(), 0);
    auto const it_max_point_reach = v_max_point_reach.begin();
    exa::for_each(0, v_point_coord_id.size(), [=]
#ifdef CUDA_ON
            __device__
#endif
    (auto const &i) -> void {
        // check if this point has already been classified as a core
        int max = 0;
        for (int ci = 0; ci < it_point_cell_reach_size[i]; ++ci) {
            int const c = it_point_cells_in_reach[it_point_cell_reach_offset[i] + ci];
            max += it_coord_cell_size[c];
        }
        it_max_point_reach[i] = max;
    });

//    auto cnt = exa::count_if(v_max_point_reach, 0, v_max_point_reach.size(), [=](auto const &v) -> bool {
//       return v < _m;
//    });
//    std::cout << "skip cnt: " << cnt << std::endl;

    // Finna cores
    exa::for_each(0, v_point_coord_id.size(), [=]
#ifdef CUDA_ON
            __device__
#endif
    (auto const &i) -> void {
        if (it_max_point_reach[i] < _m) return;
        // check if this point has already been classified as a core
        int hits = 0;
        float length;
        for (int ci = 0; ci < it_point_cell_reach_size[i]; ++ci) {
            int const c = it_point_cells_in_reach[it_point_cell_reach_offset[i] + ci];
            for (int j = 0; j < it_coord_cell_size[c]; ++j) {
                int const id2 = it_coord_id[it_coord_cell_offset[c] + j];
                // check outer reach only if necessary
                length = 0;
                for (int d = 0; d < _n_dim; ++d) {
                    length += (it_point_data[i * _n_dim + d] - it_coord_data[id2 * _n_dim + d]) *
                              (it_point_data[i * _n_dim + d] - it_coord_data[id2 * _n_dim + d]);
                }
                if (length <= _e2 && ++hits == _m) {
                    it_point_nn[i] = _m;
                    if (it_point_id[i] >= 0) {
                        it_coord_nn[it_point_id[i]] = it_point_nn[i];
                    }
                    return;
                }
            }
        }
    });

    // TODO move these allocations to parent
    d_vec<int> v_point_iota(v_point_coord_id.size());
    d_vec<int> v_point_core_id(v_point_coord_id.size());
    auto const it_point_core_id = v_point_core_id.begin();
    exa::iota(v_point_iota, 0, v_point_iota.size(), 0);
    // henda rest
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
    // setja nýtt core cluster label ef þörf er á
    // TODO move these allocations to parent
    d_vec<int> v_point_cluster_index(v_point_core_id.size(), INT32_MAX);
    auto const it_point_cluster_index = v_point_cluster_index.begin();
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
            } else {
                it_point_cluster_index[k] = it_coord_cluster_index[it_point_id[i]];
            }
        }
    });
    // TODO MPI MERGE

    // count the new labels
    auto new_cluster_cores = exa::count_if(v_point_new_cluster_mark, 0, v_point_new_cluster_mark.size(), []
#ifdef CUDA_ON
            __device__
#endif
    (auto const &v) -> bool {
        return v == 1;
    });
#ifdef DEBUG_ON
//    std::cout << "new cluster cores: " << new_cluster_cores << std::endl;
#endif
    int cluster_index_begin = static_cast<int>(v_cluster_label.size());
    v_cluster_label.resize(cluster_index_begin + new_cluster_cores);
    // Create new label ids
    exa::iota(v_cluster_label, cluster_index_begin, v_cluster_label.size(), cluster_index_begin);
    // the new label indexes
    exa::exclusive_scan(v_point_new_cluster_mark, 0, v_point_new_cluster_mark.size(), v_point_new_cluster_offset,
            0, cluster_index_begin);
    auto const it_point_new_cluster_offset = v_point_new_cluster_offset.begin();
    auto const _cluster_label_size = v_cluster_label.size();
    exa::for_each(0, v_point_core_id.size(), [=]
#ifdef CUDA_ON
        __device__
#endif
    (auto const &k) -> void {
        auto i = it_point_core_id[k];
        if (it_point_new_cluster_mark[k] == 1) {
            // mark the new cluster indexes
            it_point_cluster_index[k] = it_point_new_cluster_offset[k];
            if (it_point_id[i] >= 0) {
                it_coord_cluster_index[it_point_id[i]] = it_point_cluster_index[k];
            }
        }
    });
    auto const it_cluster_label = v_cluster_label.begin();
    d_vec<int> v_running(1);
    auto const it_running = v_running.begin();
    d_vec<int> v_running2(1);
    auto const it_running2 = v_running2.begin();
    int iter_cnt = 0;
    d_vec<int> v_point_cluster_other(v_point_core_id.size());
    auto const it_point_cluster_other = v_point_cluster_other.begin();

    do {
        ++iter_cnt;
        v_running[0] = 0;
        exa::for_each(0, v_point_core_id.size(), [=]
#ifdef CUDA_ON
        __device__
#endif
        (auto const &k) -> void {
            auto i = it_point_core_id[k];
            it_point_cluster_other[k] = it_cluster_label[it_point_cluster_index[k]];
            for (int ci = 0; ci < it_point_cell_reach_size[i]; ++ci) {
                int const c = it_point_cells_in_reach[it_point_cell_reach_offset[i] + ci];
                float length;
                for (int j = 0; j < it_coord_cell_size[c]; ++j) {
                    int const id2 = it_coord_id[it_coord_cell_offset[c] + j];
                    if (it_coord_nn[id2] < _m) continue;
                    if (it_cluster_label[it_coord_cluster_index[id2]] != it_point_cluster_other[k]) {
                        length = 0;
                        for (int d = 0; d < _n_dim; ++d) {
                            length += (it_point_data[i * _n_dim + d] - it_coord_data[id2 * _n_dim + d]) *
                                      (it_point_data[i * _n_dim + d] - it_coord_data[id2 * _n_dim + d]);
                        }
                        if (length <= _e2) {
                            it_point_cluster_other[k] = it_cluster_label[it_coord_cluster_index[id2]];
                            it_running[0] = 1;
                            return;
                        }
                    }
                }
            }
        });

        if (v_running[0] > 0) {
            int inner_iter = 0;
            // merge
            do {
                ++inner_iter;
                v_running2[0] = 0;
                exa::for_each(0, v_point_core_id.size(), [=]
#ifdef CUDA_ON
                __device__
#endif
                (auto const &k) -> void {
                    if (it_cluster_label[it_point_cluster_index[k]] < it_cluster_label[it_point_cluster_other[k]]) {
                        exa::atomic_min(&it_cluster_label[it_point_cluster_other[k]],
                                it_cluster_label[it_point_cluster_index[k]]);
                        it_running2[0] = 1;
                    } else if (it_cluster_label[it_point_cluster_index[k]] >
                               it_cluster_label[it_point_cluster_other[k]]) {
                        exa::atomic_min(&it_cluster_label[it_point_cluster_index[k]],
                                it_cluster_label[it_point_cluster_other[k]]);
                        it_running2[0] = 1;
                    }
                });
            } while (v_running2[0] > 0);

            // flatten
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
            exa::for_each(0, v_point_cluster_index.size(), [=]
#ifdef CUDA_ON
                __device__
#endif
            (auto const &k) -> void {
                while (it_cluster_label[it_point_cluster_index[k]] != it_point_cluster_index[k]) {
                    it_point_cluster_index[k] = it_cluster_label[it_point_cluster_index[k]];
                }
            });
        }
    } while (v_running[0] > 0);
    // mark all the non cores
    exa::for_each(0, v_point_core_id.size(), [=]
#ifdef CUDA_ON
            __device__
#endif
    (auto const &k) -> void {
        auto i = it_point_core_id[k];
        for (int ci = 0; ci < it_point_cell_reach_size[i]; ++ci) {
            int const c = it_point_cells_in_reach[it_point_cell_reach_offset[i] + ci];
            float length;
            for (int j = 0; j < it_coord_cell_size[c]; ++j) {
                int const id2 = it_coord_id[it_coord_cell_offset[c] + j];
                if (it_coord_nn[id2] >= _m) continue;
                if (it_coord_cluster_index[id2] == NO_CLUSTER) {
                    length = 0;
                    for (int d = 0; d < _n_dim; ++d) {
                        length += (it_point_data[i * _n_dim + d] - it_coord_data[id2 * _n_dim + d]) *
                                  (it_point_data[i * _n_dim + d] - it_coord_data[id2 * _n_dim + d]);
                    }
                    if (length <= _e2) {
                        exa::atomic_min(&it_coord_cluster_index[id2],
                                it_cluster_label[it_point_cluster_index[k]]);
                    }
                }
            }
        }
    });
     */
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
        return v == INT32_MAX;
    });

    processed = exa::count_if(v_coord_status, 0, v_coord_status.size(), [=]
#ifdef CUDA_ON
            __device__
#endif
    (int const &v) -> bool {
        return v != -2; // -2 is NOT PROCESSED
    });

#ifdef MPI_ON
    d_vec<int> v_data(2);
    v_data[0] = cores;
    v_data[1] = noise;
    mpi.allReduce(v_data, magmaMPI::sum);
    cores = v_data[0];
    noise = v_data[1];
#endif

//    v_cluster_label
    d_vec<int> v_coord_cluster_cpy = v_coord_cluster_index;
    exa::sort(v_coord_cluster_cpy, 0, v_coord_cluster_cpy.size(), []
#ifdef CUDA_ON
    __device__
#endif
    (int const &v1, int const &v2) -> bool {
        return v1 < v2;
    });
    d_vec<int> v_iota(v_coord_cluster_index.size());
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

    /*
    std::unordered_map<int, int> v_cluster_map;
    for (int const &cluster : v_coord_cluster_index) {
        if (cluster < 0) {
            std::cerr << "LESS THAN ZERO" << std::endl;
            exit(-1);
        }
        if (cluster != INT32_MAX) {
            if (v_cluster_label[cluster] != cluster) {
                std::cerr << "ERROR INCONSISTENT" << std::endl;
                exit(-1);
            }
            auto elem = v_cluster_map.find(v_cluster_label[cluster]);
            if (elem == v_cluster_map.end()) {
                v_cluster_map.insert(std::make_pair(v_cluster_label[cluster], 1));
            } else {
                (*elem).second++;
            }
        }
    }
    std::cout << "Map size: " << v_cluster_map.size() << std::endl;
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
    exa::exclusive_scan(v_node_cluster_size, 0, v_node_cluster_size.size(), v_node_cluster_offset, 0, 0);
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
    /*
    v_coord_nn.resize(n_coord, 0);
    d_vec<int> v_coord_id_sorted(n_coord);
    auto const it_coord_id_sorted = v_coord_id_sorted.begin();
    auto const _n_dim = n_dim;
    auto const _e_i = e / 2;//sqrtf(2);
    auto const _m = m;
    auto const _e = e;
    auto const _e2 = e2;
    auto const it_coord = v_coord.begin();
    auto const it_min_bound = v_min_bounds.begin();
    auto const it_coord_nn = v_coord_nn.begin();

    exa::iota(v_coord_id_sorted, 0, v_coord_id_sorted.size(), 0);
    exa::sort(v_coord_id_sorted, 0, v_coord_id_sorted.size(), [=]
#ifdef CUDA_ON
    __device__
#endif
    (auto const &i, auto &j) -> bool {
        for (int d = 0; d < _n_dim; ++d) {
            if ((int)((it_coord[i * _n_dim + d] - it_min_bound[d]) / _e_i) < (int)((it_coord[j * _n_dim + d] - it_min_bound[d]) / _e_i)) {
                return true;
            } else if ((int)((it_coord[i * _n_dim + d] - it_min_bound[d]) / _e_i) > (int)((it_coord[j * _n_dim + d] - it_min_bound[d]) / _e_i)) {
                return false;
            }
        }
        return false;
    });

//    print_cuda_memory_usage();
    d_vec<int> v_point_id(n_coord);
    exa::iota(v_point_id, 0, v_point_id.size(), 0);
    v_coord_cell_offset.resize(v_point_id.size());
    v_coord_cell_offset[0] = 0;
    exa::copy_if(v_point_id, 1, v_point_id.size(), v_coord_cell_offset, 1,[=]
#ifdef CUDA_ON
        __device__
#endif
    (auto const &i) -> bool {
        for (int d = 0; d < _n_dim; ++d) {
            if ((int)((it_coord[it_coord_id_sorted[i] * _n_dim + d] - it_min_bound[d]) / _e_i) != (int)((it_coord[it_coord_id_sorted[i - 1] * _n_dim + d] - it_min_bound[d]) / _e_i))
                return true;
        }
        return false;
    });
    v_coord_cell_size.resize(v_coord_cell_offset.size());
    auto const it_coord_cell_size = v_coord_cell_size.begin();
    auto const it_coord_cell_offset = v_coord_cell_offset.begin();
    exa::for_each(0, v_coord_cell_size.size() - 1, [=]
#ifdef CUDA_ON
        __device__
#endif
    (auto const &i) -> void {
        it_coord_cell_size[i] = it_coord_cell_offset[i + 1] - it_coord_cell_offset[i];
    });
    v_coord_cell_size[v_coord_cell_size.size()-1] = n_coord - v_coord_cell_offset[v_coord_cell_size.size()-1];
    v_coord_cell_index.resize(n_coord, -1);
    auto const it_coord_cell_index = v_coord_cell_index.begin();
    std::cout << "cells #: " << v_coord_cell_offset.size() << std::endl;

    exa::for_each(0, v_coord_cell_size.size(), [=]
#ifdef CUDA_ON
        __device__
#endif
    (auto const &i) -> void {
        for (int j = 0; j < it_coord_cell_size[i]; ++j) {
            it_coord_nn[it_coord_id_sorted[it_coord_cell_offset[i] + j]] = it_coord_cell_size[i];
            it_coord_cell_index[it_coord_id_sorted[it_coord_cell_offset[i] + j]] = i;
        }
    });

    d_vec<int> v_coord_dim_0(v_coord_cell_offset.size());
    d_vec<int> v_coord_dim_1(v_coord_cell_offset.size());
    d_vec<int> v_coord_dim_2(v_coord_cell_offset.size());
    auto const it_coord_dim_0 = v_coord_dim_0.begin();
    auto const it_coord_dim_1 = v_coord_dim_1.begin();
    auto const it_coord_dim_2 = v_coord_dim_2.begin();
    d_vec<int> v_coord_dim_iota(v_coord_cell_offset.size());
    exa::iota(v_coord_dim_iota, 0, v_coord_dim_iota.size(), 0);
    v_coord_dim_0[0] = 0;
    v_coord_dim_1[0] = 0;
    v_coord_dim_2[0] = 0;
    exa::copy_if(v_coord_dim_iota, 1, v_coord_dim_iota.size(), v_coord_dim_0, 1, [=]
#ifdef CUDA_ON
            __device__
#endif
    (auto const &i) -> bool {
        if ((int)((it_coord[it_coord_id_sorted[it_coord_cell_offset[i]] * _n_dim] - it_min_bound[0]) / _e_i)
            != (int)((it_coord[it_coord_id_sorted[it_coord_cell_offset[i - 1]] * _n_dim] - it_min_bound[0]) / _e_i))
            return true;
        return false;
    });
    exa::copy_if(v_coord_dim_iota, 1, v_coord_dim_iota.size(), v_coord_dim_1, 1, [=]
#ifdef CUDA_ON
            __device__
#endif
    (auto const &i) -> bool {
        if ((int)((it_coord[it_coord_id_sorted[it_coord_cell_offset[i]] * _n_dim] - it_min_bound[0]) / _e_i)
            != (int)((it_coord[it_coord_id_sorted[it_coord_cell_offset[i - 1]] * _n_dim] - it_min_bound[0]) / _e_i))
            return true;
        if ((int)((it_coord[it_coord_id_sorted[it_coord_cell_offset[i]] * _n_dim + 1] - it_min_bound[1]) / _e_i)
            != (int)((it_coord[it_coord_id_sorted[it_coord_cell_offset[i - 1]] * _n_dim + 1] - it_min_bound[1]) / _e_i))
            return true;
        return false;
    });

    exa::copy_if(v_coord_dim_iota, 1, v_coord_dim_iota.size(), v_coord_dim_2, 1, [=]
#ifdef CUDA_ON
            __device__
#endif
    (auto const &i) -> bool {
        if ((int)((it_coord[it_coord_id_sorted[it_coord_cell_offset[i]] * _n_dim] - it_min_bound[0]) / _e_i)
            != (int)((it_coord[it_coord_id_sorted[it_coord_cell_offset[i - 1]] * _n_dim] - it_min_bound[0]) / _e_i))
            return true;
        if ((int)((it_coord[it_coord_id_sorted[it_coord_cell_offset[i]] * _n_dim + 1] - it_min_bound[1]) / _e_i)
            != (int)((it_coord[it_coord_id_sorted[it_coord_cell_offset[i - 1]] * _n_dim + 1] - it_min_bound[1]) / _e_i))
            return true;
        if ((int)((it_coord[it_coord_id_sorted[it_coord_cell_offset[i]] * _n_dim + 2] - it_min_bound[2]) / _e_i)
            != (int)((it_coord[it_coord_id_sorted[it_coord_cell_offset[i - 1]] * _n_dim + 2] - it_min_bound[2]) / _e_i))
            return true;
        return false;
    });

    std::cout << "v_coord_dim_0 size: " << v_coord_dim_0.size() << std::endl;
    std::cout << "v_coord_dim_1 size: " << v_coord_dim_1.size() << std::endl;
    std::cout << "v_coord_dim_2 size: " << v_coord_dim_2.size() << std::endl;
//    magma_util::print_v("v_coord_dim_0: ", &v_coord_dim_0[0], v_coord_dim_0.size());

    d_vec<int> v_point_range_stack(n_coord * 2, -1);
    auto const it_range_stack = v_point_range_stack.begin();
    exa::for_each(0, n_coord, [=]
    #ifdef CUDA_ON
        __device__
    #endif
    (auto const &i) -> void {
        int hit = it_coord_nn[i];
        if (hit >= _m)
            return;

        int begin1 = std::lower_bound(v_coord_dim_0.begin(),
                v_coord_dim_0.end(),
                (int)((it_coord[i * _n_dim] - it_min_bound[0]) / _e_i),
                [=] (auto const &i2, auto const &v) -> bool {
                    if ((it_coord[it_coord_id_sorted[it_coord_cell_offset[i2]] * _n_dim] - it_min_bound[0]) / _e_i < v - 2) {
                        return true;
                    }
                    return false;
                }) - v_coord_dim_0.begin();

        int end1 = std::upper_bound(v_coord_dim_0.begin() + begin1,
                v_coord_dim_0.end(),
                (int)((it_coord[i * _n_dim] - it_min_bound[0]) / _e_i),
                [=] (auto const &v, auto const &i2) -> bool {
                    if ((it_coord[it_coord_id_sorted[it_coord_cell_offset[i2]] * _n_dim] - it_min_bound[0]) / _e_i > v + 3) {
                        return true;
                    }
                    return false;
                }) - v_coord_dim_0.begin();

        begin1 = it_coord_dim_0[begin1];
        end1 = it_coord_dim_0[end1];
//        std::cout << "1: " << begin1 << " " << end1 << std::endl;

        for (int j = begin1; j < end1; ++j) {
            // each cell has a single limit

            int begin2 = std::lower_bound(v_coord_dim_1.begin(),
                    v_coord_dim_1.end(),
                    (int)((it_coord[i * _n_dim + 1] - it_min_bound[1]) / _e_i),
                    [=] (auto const &i2, auto const &v) -> bool {
                        if ((it_coord[it_coord_id_sorted[it_coord_cell_offset[i2]] * _n_dim + 1] - it_min_bound[1]) / _e_i < v - 2) {
                            return true;
                        }
                        return false;
                    }) - v_coord_dim_1.begin();

            int end2 = std::upper_bound(v_coord_dim_1.begin(),
                    v_coord_dim_1.end(),
                    (int)((it_coord[i * _n_dim + 1] - it_min_bound[1]) / _e_i),
                    [=] (auto const &v, auto const &i2) -> bool {
                        if ((it_coord[it_coord_id_sorted[it_coord_cell_offset[i2]] * _n_dim + 1] - it_min_bound[1]) / _e_i > v + 3) {
                            return true;
                        }
                        return false;
                    }) - v_coord_dim_1.begin();

//            std::cout << "2: " << begin2 << " " << end2 << std::endl;

            begin2 = it_coord_dim_1[begin2];
            end2 = it_coord_dim_1[end2];
            for (int jj = begin2; jj < end2; ++jj) {
                if (it_coord_cell_index[i] == jj)
                    continue;

                int coord_id = it_coord_cell_offset[jj];
                for (int k = 0; k < it_coord_cell_size[jj]; ++k, ++coord_id) {
                    float length = 0;
                    for (int d = 0; d < _n_dim; ++d) {
                        length += (it_coord[i * _n_dim + d] - it_coord[it_coord_id_sorted[coord_id] * _n_dim + d]) *
                                  (it_coord[i * _n_dim + d] - it_coord[it_coord_id_sorted[coord_id] * _n_dim + d]);
                        if (length > _e2) {
                            break;
                        }
                    }
                    if (length <= _e2) {
                        if (++hit == _m) {
                            it_coord_nn[i] = _m;
                            return;
                        }
                    }
                }
            }
//            std::cout << "done" << std:: endl;




//            if (it_coord_cell_index[i] == j)
//                continue;
//
//            int coord_id = it_coord_cell_offset[j];
//            for (int k = 0; k < it_coord_cell_size[j]; ++k, ++coord_id) {
//                float length = 0;
//                for (int d = 0; d < _n_dim; ++d) {
//                    length += (it_coord[i * _n_dim + d] - it_coord[it_coord_id_sorted[coord_id] * _n_dim + d]) *
//                              (it_coord[i * _n_dim + d] - it_coord[it_coord_id_sorted[coord_id] * _n_dim + d]);
//                    if (length > _e2) {
//                        break;
//                    }
//                }
//                if (length <= _e2) {
//                    if (++hit == _m) {
//                        it_coord_nn[i] = _m;
//                        return;
//                    }
//                }
//            }
        }
//        std::cout << "done" << std::endl;
    });

    */
        /*
        int begin = std::lower_bound(v_coord_cell_offset.begin(),
                v_coord_cell_offset.end(),
                (int)((it_coord[i * _n_dim] - it_min_bound[0]) / _e_i),
                [=] (auto const &j, auto const &v) -> bool {
                    if ((it_coord[it_coord_id_sorted[j] * _n_dim] - it_min_bound[0]) / _e_i < v - 2) {
                        return true;
                    }
                    return false;
                }) - v_coord_cell_offset.begin();

        int end = std::upper_bound(v_coord_cell_offset.begin() + begin,
                v_coord_cell_offset.end(),
                (int)((it_coord[i * _n_dim] - it_min_bound[0]) / _e_i),
                [=] (auto const &v, auto const &j) -> bool {
                    if ((it_coord[it_coord_id_sorted[j] * _n_dim] - it_min_bound[0]) / _e_i > v + 3) {
                        return true;
                    }
                    return false;
                }) - v_coord_cell_offset.begin();

        int d_s = 1;
        it_range_stack[i * 2] = begin;
        it_range_stack[i * 2 + 1] = end;


        while (d_s > 0) {
//            std::cout << "low 1" << std::endl;
            begin = std::lower_bound(v_coord_cell_offset.begin() + it_range_stack[i * 2],
                    v_coord_cell_offset.begin() + it_range_stack[i * 2 + 1],
                    (int)((it_coord[i * _n_dim + d_s] - it_min_bound[d_s]) / _e_i),
                    [=] (auto const &j, auto const &v) -> bool {
                        if ((it_coord[it_coord_id_sorted[j] * _n_dim + d_s] - it_min_bound[d_s]) / _e_i < v - 5) {
                            return true;
                        }
                        return false;
                    }) - v_coord_cell_offset.begin();
//            std::cout << "low 2" << std::endl;
            end = std::upper_bound(v_coord_cell_offset.begin() + begin,
                    v_coord_cell_offset.begin() + it_range_stack[i * 2 + 1],
                    (int) ((it_coord[i * _n_dim + d_s] - it_min_bound[d_s]) / _e_i),
                    [=] (auto const &v, auto const &j) -> bool {
//                    [=] (auto const &j, auto const &v) -> bool {
                        if ((it_coord[it_coord_id_sorted[j] * _n_dim + d_s] - it_min_bound[d_s]) / _e_i > v + 5) {
                            return true;
                        }
                        return false;
                    }) - v_coord_cell_offset.begin() + 1;
//                #pragma omp critical
//            std::cout << "limit: " << begin << " : " << end << " : " << it_range_stack[i * 2 + 1] << std::endl;
//            if (end >= it_range_stack[i * 2 + 1]) {
//                --d_s;
//                continue;
//            }
//            if (begin == end) {
//                --d_s;
//                continue;
//            }
            it_range_stack[i * 2] = end;
//                if (end > v_coord_cell_offset.size())
//                    std::cerr << "ERROR end: " << end << std::endl;
//                assert(begin >= 0 && end <= v_coord_cell_offset.size());

            for (int j = begin; j < end; ++j) {
                if (it_coord_cell_index[i] == j)
                    continue;
                int coord_id = it_coord_cell_offset[j];
                for (int k = 0; k < it_coord_cell_size[j]; ++k, ++coord_id) {
                    float length = 0;
                    for (int d = 0; d < _n_dim; ++d) {
                        length += (it_coord[i * _n_dim + d] - it_coord[it_coord_id_sorted[coord_id] * _n_dim + d]) *
                                  (it_coord[i * _n_dim + d] - it_coord[it_coord_id_sorted[coord_id] * _n_dim + d]);
                        if (length > _e2) {
                            break;
                        }
                    }
                    if (length <= _e2) {
                        if (++hit == _m) {
                            it_coord_nn[i] = _m;
                            return;
                        }
                    }
                }
            }
            return;
        }

    });
*/
    /*
    int cell_cnt = 0;

    exa::for_each(0, n_coord, [&]
#ifdef CUDA_ON
            __device__
#endif
    (auto const &i) -> void {
        int hit = it_coord_nn[i];
        if (hit >= _m)
            return;
        int d = 0;
        int begin = std::lower_bound(v_coord_cell_offset.begin(),
                v_coord_cell_offset.end(),
                i,
                [=] (auto const &j, auto const &v) -> bool {
            if (it_coord[it_coord_id_sorted[j] * _n_dim + d] < it_coord[v * _n_dim + d] - _e - _e_i) {
                return true;
            }
            return false;
        }) - it_coord_cell_offset;
        int end = std::upper_bound(v_coord_cell_offset.begin(),
                v_coord_cell_offset.end(),
                i,
                [=] (auto const &v, auto const &j) -> bool {
                    if (it_coord[it_coord_id_sorted[j] * _n_dim + d] > it_coord[v * _n_dim + d] + _e + _e_i) {
                        return true;
                    }
                    return false;
                }) - it_coord_cell_offset;


        #pragma omp atomic
        cell_cnt += end - begin;

        for (int j = begin; j < end; ++j) {
            if (it_coord_cell_index[i] == j)
                continue;
            int coord_id = it_coord_cell_offset[j];

            for (int k = 0; k < it_coord_cell_size[j]; ++k, ++coord_id) {
                float length = 0;
                for (d = 0; d < _n_dim; ++d) {
                    length += (it_coord[i * _n_dim + d] - it_coord[it_coord_id_sorted[coord_id] * _n_dim + d]) *
                              (it_coord[i * _n_dim + d] - it_coord[it_coord_id_sorted[coord_id] * _n_dim + d]);
                    if (length > _e2) {
                        break;
                    }
                }
                if (length <= _e2) {
                    if (++hit == _m) {
                        it_coord_nn[i] = _m;
                        return;
                    }
                }
            }
        }

    });

    std::cout << "CELL CNT: " << cell_cnt << std::endl;


    */




    /*
//    for (int i = 0; i < n_coord; ++i) {
//        assert(v_coord_cell_index[i] >= 0);
//        assert(v_coord_nn[i] > 0);
//    }

//    print_cuda_memory_usage();


    d_vec<int> v_point_cell_range(n_coord * 2);
    auto const it_point_cell_range = v_point_cell_range.begin();

//    print_cuda_memory_usage();
    exa::lower_bound(v_coord_cell_offset, 0, v_coord_cell_offset.size(), v_point_id, 0, v_point_id.size(), v_point_cell_range, 0, 1, [=]
#ifdef CUDA_ON
    __device__
#endif
    (auto const &i, auto const &v) -> bool {
        for (int d = 0; d < _n_dim; ++d) {
            if (it_coord[it_coord_id_sorted[i] * _n_dim + d] < it_coord[v * _n_dim + d] - _e - _e_i)
                return true;
            if (it_coord[it_coord_id_sorted[i] * _n_dim + d] > it_coord[v * _n_dim + d] - _e - _e_i)
                return false;
        }
        return false;
    });

    exa::upper_bound(v_coord_cell_offset, 0, v_coord_cell_offset.size(), v_point_id, 0, v_point_id.size(), v_point_cell_range, 1, 1, [=]
#ifdef CUDA_ON
            __device__
#endif
    (auto const &v, auto const &i) -> bool {
        for (int d = 0; d < _n_dim; ++d) {
            if (it_coord[it_coord_id_sorted[i] * _n_dim + d] > it_coord[v * _n_dim + d] + _e + _e_i)
                return true;
            if (it_coord[it_coord_id_sorted[i] * _n_dim + d] < it_coord[v * _n_dim + d] + _e + _e_i)
                return false;
        }
        return false;
    });

    exa::for_each(0, n_coord, [=]
#ifdef CUDA_ON
        __device__
#endif
    (auto const &i) -> void {
        int hit = it_coord_nn[i];
        if (hit >= _m)
            return;

        int begin = it_point_cell_range[i * 2];
        int end = it_point_cell_range[i * 2 + 1];

        for (int j = begin; j < end; ++j) {
            if (it_coord_cell_index[i] == j)
                continue;
            int coord_id = it_coord_cell_offset[j];

            bool is_ok = true;
            for (int d = 0; d < _n_dim; ++d) {
                if (it_coord[i * _n_dim + d] + _e + _e_i < it_coord[it_coord_id_sorted[coord_id] * _n_dim + d]) {
                    is_ok = false;
                    break;
                }
                if (it_coord[i * _n_dim + d] - _e - _e_i > it_coord[it_coord_id_sorted[coord_id] * _n_dim + d]) {
                    is_ok = false;
                    break;
                }
            }
            if (!is_ok)
                continue;

            for (int k = 0; k < it_coord_cell_size[j]; ++k, ++coord_id) {
                float length = 0;
                for (int d = 0; d < _n_dim; ++d) {
                    length += (it_coord[i * _n_dim + d] - it_coord[it_coord_id_sorted[coord_id] * _n_dim + d]) *
                              (it_coord[i * _n_dim + d] - it_coord[it_coord_id_sorted[coord_id] * _n_dim + d]);
                    if (length > _e2) {
                        break;
                    }
                }
                if (length <= _e2) {
                    if (++hit == _m) {
                        it_coord_nn[i] = _m;
                        return;
                    }
                }
            }
        }
    });
    */
    /*
    int cores = exa::count_if(v_coord_nn, 0, v_coord_nn.size(), [=]
#ifdef CUDA_ON
            __device__
#endif
    (auto const &v) -> bool {
        return v >= _m;
    });

    std::cout << "CORES: " << cores << std::endl;
    */
}
    /*
    // Initialize
    v_coord_nn.resize(n_coord, 0);
    auto const it_coord_nn = v_coord_nn.begin();
    d_vec<int> v_coord_dim_index(n_dim * n_coord);
    auto const it_coord_dim_index = v_coord_dim_index.begin();
    d_vec<int> v_coord_dim_index_size(n_dim, n_coord);
    d_vec<int> v_coord_dim_index_offset(n_dim);
    exa::exclusive_scan(v_coord_dim_index_size, 0, v_coord_dim_index_size.size(), v_coord_dim_index_offset, 0, 0);
    auto const it_coord_dim_index_offset = v_coord_dim_index_offset.begin();
    d_vec<float> v_coord_dim_sorted(n_dim * n_coord, FLOAT_MAX);
    auto const it_coord_dim_sorted = v_coord_dim_sorted.begin();
    d_vec<int> v_coord_dim_range(n_dim * n_coord * 4);
    auto const it_coord_dim_range = v_coord_dim_range.begin();
    d_vec<float> v_search_value(n_coord);
    auto const it_search_value = v_search_value.begin();
    auto const it_coord = v_coord.begin();
    auto const _n_dim = n_dim;
    auto const _e = e;
    auto const _e2 = e2;
    auto const _ei2 = ((e / sqrtf(2)) / 2) * ((e / sqrtf(2)) / 2);
    auto const _m = m;
    auto const _esr2 = (e / sqrtf(2)) / 2;

    std::cout << "e: " << e << " m: " << m << std::endl;

    // sort the fuckers
    for (int d = 0; d < n_dim; ++d) {
        exa::iota(v_coord_dim_index, v_coord_dim_index_offset[d], v_coord_dim_index_offset[d] + v_coord_dim_index_size[d], 0);
        exa::sort(v_coord_dim_index, v_coord_dim_index_offset[d], v_coord_dim_index_offset[d] + v_coord_dim_index_size[d], [=]
#ifdef CUDA_ON
    __device__
#endif
        (auto const &i1, auto const &i2) {
            return it_coord[i1 * _n_dim + d] < it_coord[i2 * _n_dim + d];
        });
    }

    // fill the sorted coords
    exa::for_each(0, n_coord, [=]
#ifdef CUDA_ON
        __device__
#endif
    (auto const &i) -> void {
        for (int d = 0; d < _n_dim; ++d) {
            it_coord_dim_sorted[it_coord_dim_index_offset[d] + i] = it_coord[it_coord_dim_index[it_coord_dim_index_offset[d] + i] * _n_dim + d];
        }
    });

    int stride = static_cast<int>((n_dim * 4)) - 1;
    int out_begin = 0;
    // search the fuckers
    for (int d = 0; d < n_dim; ++d) {

        exa::for_each(0, n_coord, [=]
#ifdef CUDA_ON
        __device__
#endif
        (auto const &i) -> void {
            it_search_value[i] = it_coord[i * _n_dim + d] - _e;
        });

//        std::cout << "search values low: " << _e << " ";
//        for (int i = 0; i < 10; ++i) {
//            std::cout << v_search_value[i] << " ";
//        }
//        std::cout << std::endl;

        exa::lower_bound(v_coord_dim_sorted, v_coord_dim_index_offset[d],
                v_coord_dim_index_offset[d] + v_coord_dim_index_size[d], v_search_value, 0,
                v_search_value.size(), v_coord_dim_range, out_begin++, stride);

        exa::for_each(0, n_coord, [=]
#ifdef CUDA_ON
        __device__
#endif
        (auto const &i) -> void {
            it_search_value[i] = it_search_value[i] + _e + _e;
        });

        exa::upper_bound(v_coord_dim_sorted, v_coord_dim_index_offset[d],
                v_coord_dim_index_offset[d] + v_coord_dim_index_size[d], v_search_value, 0,
                v_search_value.size(), v_coord_dim_range, out_begin++, stride);

        exa::for_each(0, n_coord, [=]
#ifdef CUDA_ON
            __device__
#endif
        (auto const &i) -> void {
            it_search_value[i] = it_coord[i * _n_dim + d] - _esr2;
        });
        // TODO tighten search bound
        exa::lower_bound(v_coord_dim_sorted, v_coord_dim_index_offset[d],
                v_coord_dim_index_offset[d] + v_coord_dim_index_size[d], v_search_value, 0,
                v_search_value.size(), v_coord_dim_range, out_begin++, stride);
        exa::for_each(0, n_coord, [=]
#ifdef CUDA_ON
                __device__
#endif
        (auto const &i) -> void {
            it_search_value[i] = it_coord[i * _n_dim + d] + _esr2;
        });
        // TODO tighten search bound
        exa::upper_bound(v_coord_dim_sorted, v_coord_dim_index_offset[d],
                v_coord_dim_index_offset[d] + v_coord_dim_index_size[d], v_search_value, 0,
                v_search_value.size(), v_coord_dim_range, out_begin++, stride);
    }

    // pick the best dimension
    d_vec<int> v_best_range(n_coord * 4);
    auto const it_best_range = v_best_range.begin();
    exa::for_each(0, n_coord, [=]
#ifdef CUDA_ON
            __device__
#endif
    (auto const &i) -> void {
        int s_d = 0;
        int range_begin = it_coord_dim_range[(i * _n_dim * 4)];
        int range_end = it_coord_dim_range[(i * _n_dim * 4) + 1];
        int size = range_end - range_begin;

        // find the shortest range
        for (int d = 1; d < _n_dim; ++d) {
            if (it_coord_dim_range[(i * _n_dim * 4) + (d * 4) + 1] - it_coord_dim_range[(i * _n_dim * 4) + (d * 4)] < size) {
                s_d = d;
                range_begin = it_coord_dim_range[(i * _n_dim * 4) + (d * 4)];
                range_end = it_coord_dim_range[(i * _n_dim * 4) + (d * 4) + 1];
                size = range_end - range_begin;
            }
        }
        it_best_range[i*4] = range_begin + it_coord_dim_index_offset[s_d];
        it_best_range[i*4+1] = range_end + it_coord_dim_index_offset[s_d];
        it_best_range[i*4+2] = it_coord_dim_range[(i * _n_dim * 4) + 2] + it_coord_dim_index_offset[s_d];
        it_best_range[i*4+3] = it_coord_dim_range[(i * _n_dim * 4) + 3] + it_coord_dim_index_offset[s_d];
    });

    d_vec<int> v_hit_table(n_coord, 0);
    auto const it_hit_table = v_hit_table.begin();

    for (std::size_t i = 0; i < n_coord; ++i) {
//        exa::for_each(0, n_coord, [=]
        exa::for_each(v_best_range[i*4], v_best_range[i*4+1], [=]
#ifdef CUDA_ON
        __device__
#endif
        (auto const &j) -> void {
            float length = 0;
            for (int d = 0; d < _n_dim; ++d) {
                length += (it_coord[i * _n_dim + d] - it_coord[it_coord_dim_index[j] * _n_dim + d]) *
                          (it_coord[i * _n_dim + d] - it_coord[it_coord_dim_index[j] * _n_dim + d]);
                if (length > _e2) {
                    break;
                }
            }
            if (length <= _e2) {
                it_hit_table[it_coord_dim_index[j]] = 1;
            }
        });
        it_coord_nn[i] = exa::count_if(v_hit_table, 0, n_coord, []
//        it_coord_nn[i] = exa::count_if(v_hit_table, v_best_range[i*4], v_best_range[i*4+1], []
#ifdef CUDA_ON
    __device__
#endif
        (auto const &v) -> bool {
            return v > 0;
        });
//        exa::fill(v_hit_table, v_best_range[i*4], v_best_range[i*4+1], 0);
        exa::fill(v_hit_table, 0, n_coord, 0);
    }


    int cnt = exa::count_if(v_coord_nn, 0, v_coord_nn.size(), [=]
#ifdef CUDA_ON
            __device__
#endif
    (auto const &v) -> bool {
        return v >= _m;
    });

    std::cout << "CORES: " << cnt << std::endl;

}
    */





    /*

    // process the outer

    // Count the fuckers
    exa::for_each(0, n_coord, [=]
#ifdef CUDA_ON
        __device__
#endif
    (auto const &i) -> void {
        if (it_coord_nn[i] >= _m) {
            return;
        }
        int range_begin = it_best_range[(i * 4)];
        int range_end = it_best_range[(i * 4) + 1];
        int range_begin_inner = it_best_range[(i * 4) + 2];
        int range_end_inner = it_best_range[(i * 4) + 3];
        int size = range_end - range_begin;
        if (size < _m)
            return;
        int hit = 0;
        int hit_inner = 0;
        float length = 0;

        for (int j = range_begin; j < range_end; ++j, --size) {
//            if (size + hit < _m)
//                return;
            int id2 = it_coord_dim_index[j];
            length = 0;
            for (int d = 0; d < _n_dim; ++d) {
                length += (it_coord[i * _n_dim + d] - it_coord[id2 * _n_dim + d]) *
                          (it_coord[i * _n_dim + d] - it_coord[id2 * _n_dim + d]);
                if (length > _e2) {
                    break;
                }
            }
            if (length <= _e2 && ++hit == _m) {
                it_coord_nn[i] = hit;
//                return;
            }
        }

    });

    int cnt = exa::count_if(v_coord_nn, 0, v_coord_nn.size(), [=]
#ifdef CUDA_ON
    __device__
#endif
    (auto const &v) -> bool {
        return v >= _m;
    });

    std::cout << "CORES: " << cnt << std::endl;

    return;

    v_coord_nn.resize(n_coord, 1);
    v_coord_cluster_index.resize(n_coord, NO_CLUSTER);
    v_coord_status.resize(n_coord, NOT_PROCESSED);

    auto const it_coord_status = v_coord_status.begin();
    std::size_t coord_begin = 0;
    int chunk_size = 50000;
    d_vec<int> v_point_coord_id(chunk_size);
    d_vec<float> v_point_data(chunk_size * n_dim);
    d_vec<float> v_point_data_pruned;
    d_vec<long long> v_point_index(v_point_coord_id.size());
    d_vec<int> v_point_cells_in_reach(v_point_coord_id.size());
    d_vec<int> v_point_cell_reach_offset(v_point_coord_id.size());
    d_vec<int> v_point_cell_reach_size(v_point_coord_id.size());
    do {
        exa::iota(v_point_coord_id, 0, v_point_coord_id.size(), coord_begin);
        std::size_t coord_end = coord_begin + chunk_size;
        if (coord_end > n_coord) {
            coord_end = n_coord;
        }
        auto const _n_dim = n_dim;
        exa::fill(v_point_data, 0, v_point_data.size(), FLOAT_MAX);
        exa::copy(v_coord, coord_begin * n_dim, coord_end * n_dim, v_point_data, 0);
        if (exa::count_if(v_point_data, 0, v_point_data.size(), []
#ifdef CUDA_ON
        __device__
#endif
        (auto const &v) -> bool {
            return v == _FLOAT_MAX;
        }) == 0) {
            process_points3(v_point_coord_id, v_point_data, v_point_index, v_point_cells_in_reach,
                    v_point_cell_reach_offset, v_point_cell_reach_size, mpi);
        } else {
            // we need to prune the data
            v_point_data_pruned.resize(v_point_data.size());
            exa::copy_if(v_point_data, 0, v_point_data.size(), v_point_data_pruned, 0, []
#ifdef CUDA_ON
    __device__
#endif
            (auto const &v) -> bool {
                return v != _FLOAT_MAX;
            });
            v_point_coord_id.resize(v_point_data_pruned.size() / n_dim);
            v_point_cells_in_reach.resize(v_point_coord_id.size());
            v_point_cell_reach_offset.resize(v_point_coord_id.size());
            v_point_cell_reach_size.resize(v_point_coord_id.size());
            process_points3(v_point_coord_id, v_point_data_pruned, v_point_index, v_point_cells_in_reach,
                    v_point_cell_reach_offset, v_point_cell_reach_size, mpi);
        }
        coord_begin += chunk_size;
    } while (coord_begin < n_coord);
}
     */

/*
void data_process::select_and_process(magmaMPI mpi) noexcept {
    // set to 1 as every coord is connected with itself
    v_coord_nn.resize(n_coord, 1);
    v_coord_cluster_index.resize(n_coord, NO_CLUSTER);
    v_coord_status.resize(n_coord, NOT_PROCESSED);
    auto const it_coord_status = v_coord_status.begin();

    d_vec<int> v_point_id(v_coord_id.size());
#ifdef DEBUG_ON
    if (mpi.rank == 0)
        std::cout << "number of cells: " << static_cast<int>(v_coord_cell_size.size()) << std::endl;
#endif
    exa::iota(v_point_id, 0, v_point_id.size(), 0);

    // TODO update for MPI
    std::size_t n_sample_size = 5000;

//    d_vec<int> v_coord_id_index(n_coord);
//    d_vec<int> v_coord_index_marker(n_coord, 1);
//    auto const it_coord_index_marker = v_coord_index_marker.begin();
//    d_vec<int> v_coord_index_marker_offset(n_coord);

//    d_vec<int> v_unprocessed_id_2;

    /*
    d_vec<int> v_id_index(n_coord);
    auto const it_id_index = v_id_index.begin();
    d_vec<int> v_id_chunk;//(n_sample_size);
    d_vec<float> v_data_chunk;//(n_sample_size * n_dim);
    auto const it_coord_cluster_index = v_coord_cluster_index.begin();
    auto const _n_dim = n_dim;
    int n_iter = 0;
    d_vec<int> v_is_running(1, 1);
    auto const it_is_running = v_is_running.begin();
    auto const it_coord_id = v_coord_id.begin();
    auto const it_coord = v_coord.begin();
    while (v_is_running[0] == 1) {
        ++n_iter;
        std::cout << "Starting loop: " << n_iter << std::endl;
        v_is_running[0] = 0;
        v_id_index.resize(n_coord);
        exa::fill(v_id_index, 0, v_id_index.size(), INT32_MAX);
        exa::for_each(0, v_id_index.size(), [=](auto const &i) -> void {
//            if (it_coord_status[it_coord_id[i]] == MARKED
//                || (it_coord_status[it_coord_id[i]] != PROCESSED && it_coord_cluster_index[it_coord_id[i]] == NO_CLUSTER)) {
            if (it_coord_status[it_coord_id[i]] != PROCESSED) {
                it_id_index[i] = i;
                it_is_running[0] = 1;
            }
        });
        if (v_is_running[0] == 1) {
            exa::sort(v_id_index, 0, v_id_index.size(), [](auto const &i1, auto const &i2) -> bool {
               return i1 < i2;
            });
            auto pos = exa::lower_bound(v_id_index, 0, v_id_index.size(), INT32_MAX);
            v_id_index.resize(pos < n_sample_size? pos : n_sample_size);
            v_id_chunk.resize(v_id_index.size());
            v_data_chunk.resize(v_id_index.size() * n_dim);
            auto const it_id_chunk = v_id_chunk.begin();
            auto const it_data_chunk = v_data_chunk.begin();
            exa::for_each(0, v_id_index.size(), [=](auto const &i) -> void {
                it_id_chunk[i] = it_coord_id[it_id_index[i]];
                for (int j = 0; j < _n_dim; ++j) {
                    it_data_chunk[i * _n_dim + j] = it_coord[it_coord_id[it_id_index[i]] * _n_dim + j];
                }
            });
            process_points(v_id_chunk, v_data_chunk, mpi);
        }
    }
    */


    /*
    d_vec<int> v_is_running(1, 1);
    while (v_is_running[0] == 1) {
        if (n_iter++ == 0) {
            exa::iota(v_coord_id_index, 0, v_coord_id_index.size(), 0);
            exa::fill(v_coord_index_marker, 0, v_coord_index_marker.size(), 1);
        } else {
            exa::fill(v_coord_index_marker, 0, v_coord_index_marker.size(), 0);
            // update unprocessed id vector
            exa::for_each(0, v_coord_id.size(), [=](auto const &i) -> void {
                if (it_coord_status[i] == PROCESSED)
                    return;
                it_coord_index_marker[i] = 1;
            });
        }
        exa::exclusive_scan(v_coord_index_marker, 0, v_coord_index_marker.size(),
                v_coord_index_marker_offset, 0, 0);
        // select from the unprocessed ids
    }
     */


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


/*
    d_vec<int> v_id_chunk;
    d_vec<float> v_data_chunk;
    int n_blocks = 10;
    for (int i = 0; i < n_blocks; ++i) {
        int block_size = magma_util::get_block_size(i, static_cast<int>(v_point_id.size()), n_blocks);
        int block_offset = magma_util::get_block_offset(i, static_cast<int>(v_point_id.size()), n_blocks);
        v_id_chunk.resize(block_size);
        v_data_chunk.resize(block_size * n_dim);
        std::cout << "block offset: " << block_offset << " size: " << block_size << std::endl;
        std::copy(std::next(v_point_id.begin(), block_offset),
                std::next(v_point_id.begin(), block_offset+block_size),
                v_id_chunk.begin());
        std::copy(std::next(v_coord.begin(), block_offset*n_dim),
                std::next(v_coord.begin(), (block_offset+block_size)*n_dim),
                v_data_chunk.begin());
        process_points(v_id_chunk, v_data_chunk, mpi);
    }
}
 */

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

void data_process::initialize_cells() noexcept {
    v_dim_part_size.resize(2);
    v_dim_part_size[0] = (v_max_bounds[v_dim_order[0]] - v_min_bounds[v_dim_order[0]]) / e + 1;
    v_dim_part_size[1] = (v_max_bounds[v_dim_order[1]] - v_min_bounds[v_dim_order[1]]) / e + 1;
    if (static_cast<int64_t>(v_dim_part_size[0]) * v_dim_part_size[1] < 0) {
        std::cerr << "FAIL: The epsilon value is too low and therefore not supported by the current version for the"
                     " input dataset (" << static_cast<uint64_t>(v_dim_part_size[0]) * v_dim_part_size[1] << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
    auto v_iota = v_coord_id;
    d_vec<long long> v_point_cell_index(v_coord_id.size());

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
    (auto const &i) -> bool {
        return it_index[it_coord_id[i]] != it_index[it_coord_id[i - 1]];
    });
    v_coord_cell_size.resize(v_coord_cell_offset.size());
    auto const it_coord_cell_size = v_coord_cell_size.begin();
    auto const it_coord_cell_offset = v_coord_cell_offset.begin();
    exa::for_each(0, v_coord_cell_size.size() - 1, [=]
#ifdef CUDA_ON
        __device__
#endif
    (auto const &i) -> void {
        it_coord_cell_size[i] = it_coord_cell_offset[i + 1] - it_coord_cell_offset[i];
    });
    v_coord_cell_size[v_coord_cell_size.size()-1] = n_coord - v_coord_cell_offset[v_coord_cell_size.size()-1];
    v_coord_cell_index.resize(v_coord_cell_offset.size());
    auto const it_coord_cell_index = v_coord_cell_index.begin();
    auto const it_point_cell_index = v_point_cell_index.begin();
    exa::for_each(0, v_coord_cell_index.size(), [=]
#ifdef CUDA_ON
        __device__
#endif
    (auto const &i) -> void {
        it_coord_cell_index[i] = it_point_cell_index[it_coord_id[it_coord_cell_offset[i]]];
    });

//    v_coord_cell_id.resize(n_coord);

}

void data_process::build_nc_tree() noexcept {
    auto const _n_dim = n_dim;
    auto const _e_i = get_lowest_e(e, n_dim);//sqrtf(2);
    auto const _m = m;
    auto const _e = e;
    auto const _e2 = e2;
    auto const _log2 = logf(2);
    auto const it_coord = v_coord.begin();
    auto const it_min_bound = v_min_bounds.begin();

    // calculate height
    d_vec<int> v_dim_height(n_dim);
    auto const it_dim_height = v_dim_height.begin();
    auto const it_min_bounds = v_min_bounds;
    auto const it_max_bounds = v_max_bounds;
    exa::for_each(0, v_dim_height.size(), [=]
#ifdef CUDA_ON
    __device__
#endif
    (auto const &d) -> void {
        float const max_limit = it_max_bounds[d] - it_min_bounds[d];
        it_dim_height[d] = ceilf(logf(max_limit / _e_i) / _log2) + 1;
    });
    auto const nc_height = exa::minmax_element(v_dim_height, 0, v_dim_height.size(), [](auto const &v1, auto const &v2) -> bool {
       return v1 < v2;
    }).second;
    // index size and offset
    v_nc_lvl_size.resize(nc_height);
    v_nc_lvl_offset.resize(nc_height);
    auto const it_nc_lvl_size = v_nc_lvl_size.begin();
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
                        p_index = it_coord_cell_index[n_coord + v_nc_lvl_offset[l - level_mod - 1] + it_coord_cell_offset[it_nc_lvl_offset[l - level_mod] + p_index]];
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
        std::cout << "cells #: " << v_nc_lvl_size[l] << std::endl;
    }

#ifdef DEBUG_ON
    std::cout << "*****" << std::endl;
    std::cout << "COVERAGE CHECK" << std::endl;
    std::stack<int> stack;
    magma_util::print_v("v_nc_lvl_offset: ", &v_nc_lvl_offset[0], v_nc_lvl_offset.size());
    d_vec<int> v_test(n_coord);
    for (int lvl = 0; lvl < nc_height; ++lvl) {
        exa::fill(v_test, 0, v_test.size(), 0);
        std::cout << "process lvl: " << lvl << " with size: " << v_nc_lvl_size[lvl] << std::endl;

        for (int i = 0; i < v_nc_lvl_size[lvl]; ++i) {
            stack.push(lvl);
            stack.push(i);
        }
        while (!stack.empty()) {
            int c = stack.top();
            stack.pop();
            int l = stack.top();
            stack.pop();
            for (int j = 0; j < v_coord_cell_size[v_nc_lvl_offset[l] + c]; ++j) {
                if (l == 0) {
                    ++v_test[v_coord_cell_index[v_coord_cell_offset[c] + j]];
                } else {
                    stack.push(l-1);
                    stack.push(v_coord_cell_index[n_coord + v_nc_lvl_offset[l-1] + v_coord_cell_offset[v_nc_lvl_offset[l] + c] + j]);
                }
            }
        }

        bool is_fail = false;
        for (auto const &v : v_test) {
            if (v != 1) {
                is_fail = true;
                break;
            }
        }
        if (is_fail) {
            std::cout << "TEST FAILED (!!!!) at level " << lvl << std::endl;
            exit(-1);
        } else {
            std::cout << "PASSED TEST at level " << lvl << std::endl;
        }
    }
    std::cout << "*****" << std::endl;
#endif

    float const _min_float = -std::numeric_limits<float>::max();
    float const _max_float = std::numeric_limits<float>::max();

//    std::cout << "min max: " << _min_float << " " << _max_float << std::endl;

    auto const it_coord_cell_index = v_coord_cell_index.begin();
    auto const it_coord_cell_offset = v_coord_cell_offset.begin();
    // Determine the AABBs
    v_cell_AABB.resize(v_coord_cell_offset.size() * n_dim * 2, -1);
    auto const it_cell_AABB = v_cell_AABB.begin();
    for (std::size_t i = 0; i < v_coord_cell_offset.size(); ++i) {
        for (int d = 0; d < _n_dim; ++d) {
            it_cell_AABB[(i * _n_dim * 2) + d] = _max_float;
        }
        for (int d = 0; d < _n_dim; ++d) {
            it_cell_AABB[(i * _n_dim * 2) + _n_dim + d] = _min_float;
        }
    }

//    std::cout << "CHECK" << std::endl;
//    for (auto const &v : v_cell_AABB) {
//        assert(v == _max_float || v == _min_float);
//    }
//    std::cout << "CHECK END" << std::endl;


    // level 0
    for (int i = 0; i < v_nc_lvl_size[0]; ++i) {
//        if (v_coord_cell_size[i] <= 0) {
//            std::cerr << "ERROR!!!" << std::endl;
//        }
        for (int j = 0; j < v_coord_cell_size[i]; ++j) {
            auto const p = it_coord_cell_index[it_coord_cell_offset[i] + j];
            for (int d = 0; d < _n_dim; ++d) {
//                if (it_coord[p * _n_dim + d] == _max_float || it_coord[p * _n_dim + d] == _min_float) {
//                    std::cerr << "ERROR!!!" << std::endl;
//                }
//                if (j == 0)
//                    assert(it_cell_AABB[(i * _n_dim * 2) + d] == _max_float);
                if (it_coord[p * _n_dim + d] < it_cell_AABB[(i * _n_dim * 2) + d]) {
                    it_cell_AABB[(i * _n_dim * 2) + d] = it_coord[p * _n_dim + d];
                }
//                else if (j == 0) {
//                    std::cout << "ERROR!!! " << it_coord[p * _n_dim + d] << " : " << it_cell_AABB[(i * _n_dim * 2) + d] << std::endl;
//                }
//                if (j == 0)
//                    assert(it_cell_AABB[(i * _n_dim * 2) + d] != _max_float);
            }
            for (int d = 0; d < _n_dim; ++d) {
//                if (j == 0)
//                    assert(it_cell_AABB[(i * _n_dim * 2) + _n_dim + d] == _min_float);
                if (it_coord[p * _n_dim + d] > it_cell_AABB[(i * _n_dim * 2) + _n_dim + d]) {
                    it_cell_AABB[(i * _n_dim * 2) + _n_dim + d] = it_coord[p * _n_dim + d];
                }
//                else if (j == 0) {
//                    std::cout << "ERROR!!! " << it_coord[p * _n_dim + d] << " : " << it_cell_AABB[(i * _n_dim * 2) + _n_dim + d] << std::endl;
//                    std::cout << "Error ct. " << (it_coord[p * _n_dim + d] >= it_cell_AABB[(i * _n_dim * 2) + _n_dim + d]) << std::endl;
//                    std::cout << "Error ct. " << (it_coord[p * _n_dim + d] < it_cell_AABB[(i * _n_dim * 2) + _n_dim + d]) << std::endl;
//                }
//                if (j == 0)
//                    assert(it_cell_AABB[(i * _n_dim * 2) + _n_dim + d] != _min_float);
            }
        }
    }

    std::cout << "CHECK END 2" << std::endl;

    for (int i = 0; i < v_nc_lvl_size[0]; ++i) {
        for (int d = 0; d < 2 * _n_dim; ++d) {
            assert(it_cell_AABB[(i * _n_dim * 2) + d] != _min_float && it_cell_AABB[(i * _n_dim * 2) + d] != _max_float);
        }
    }

    std::cout << "CHECK END 3" << std::endl;

    for (int lvl = 1; lvl < nc_height; ++lvl) {
        for (int i = 0; i < v_nc_lvl_size[lvl]; ++i) {
            for (int j = 0; j < v_coord_cell_size[v_nc_lvl_offset[lvl] + i]; ++j) {
                auto const c = v_coord_cell_index[n_coord + v_nc_lvl_offset[lvl-1] + v_coord_cell_offset[v_nc_lvl_offset[lvl] + i] + j];
                for (int d = 0; d < _n_dim; ++d) {
                    if (it_cell_AABB[((it_nc_lvl_offset[lvl-1] + c) * _n_dim * 2) + d] < it_cell_AABB[((it_nc_lvl_offset[lvl] + i) * _n_dim * 2) + d]) {
                        it_cell_AABB[((it_nc_lvl_offset[lvl] + i) * _n_dim * 2) + d] = it_cell_AABB[((it_nc_lvl_offset[lvl-1] + c) * _n_dim * 2) + d];
                    }
                    if (it_cell_AABB[((it_nc_lvl_offset[lvl-1] + c) * _n_dim * 2) + _n_dim + d] > it_cell_AABB[((it_nc_lvl_offset[lvl] + i) * _n_dim * 2) + _n_dim + d]) {
                        it_cell_AABB[((it_nc_lvl_offset[lvl] + i) * _n_dim * 2) + _n_dim + d] = it_cell_AABB[((it_nc_lvl_offset[lvl-1] + c) * _n_dim * 2) + _n_dim + d];
                    }
                }
            }
        }
    }

//    for (int lvl = 0; lvl < nc_height; ++lvl) {
//        std::cout << "lvl: " << lvl << std::endl;
//        magma_util::print_v("min: ", &it_cell_AABB[it_nc_lvl_offset[lvl]], n_dim);
//    }



    // TEST




    /*
    for (int c1 = 0; c1 < it_nc_lvl_size[0]; ++c1) {
        for (int c2 = c1 + 1; c2 < it_nc_lvl_size[0]; ++c2) {
            bool are_connected = true;
            auto const min1 = &it_cell_AABB[((it_nc_lvl_offset[0] + c1) * _n_dim * 2)];
            auto const max1 = &it_cell_AABB[((it_nc_lvl_offset[0] + c1) * _n_dim * 2) + _n_dim];
            auto const min2 = &it_cell_AABB[((it_nc_lvl_offset[0] + c2) * _n_dim * 2)];
            auto const max2 = &it_cell_AABB[((it_nc_lvl_offset[0] + c2) * _n_dim * 2) + _n_dim];

            for (int d = 0; d < _n_dim; ++d) {
                if ((min2[d] > (max1[d] + _e) || min2[d] < (min1[d] - _e)) &&
                    (min1[d] > (max2[d] + _e) || min1[d] < (min2[d] - _e)) &&
                    (max2[d] > (max1[d] + _e) || max2[d] < (min1[d] - _e)) &&
                    (max1[d] > (max2[d] + _e) || max1[d] < (min2[d] - _e))) {
                    are_connected = false;
                    break;
                }
            }
            if (are_connected) {
                for (int i = 0; i < it_coord_cell_size[it_nc_lvl_offset[0] + c1]; ++i) {
                    int p1 = it_coord_cell_index[it_nc_lvl_offset[0] + it_coord_cell_offset[c1] + i];
                    for (int j = 0; j < it_coord_cell_size[it_nc_lvl_offset[0] + c2]; ++j) {
                        int p2 = it_coord_cell_index[it_nc_lvl_offset[0] + it_coord_cell_offset[c2] + j];
                        float length = 0;
                        for (int d = 0; d < _n_dim; ++d) {
                            length += (it_coord[p1 * _n_dim + d] - it_coord[p2 * _n_dim + d]) *
                                      (it_coord[p1 * _n_dim + d] - it_coord[p2 * _n_dim + d]);
                            if (length > _e2) {
                                break;
                            }
                        }
                        if (length <= _e2) {
                            it_coord_nn[p1] = it_coord_nn[p1] + 1;
                            it_coord_nn[p2] = it_coord_nn[p2] + 1;
                        }
                    }
                }
            }
        }
    }
     */

}

void data_process::process_local_nc_tree() noexcept {
    v_coord_nn.resize(n_coord, 0);
    v_coord_status.resize(n_coord);
    v_coord_cluster_index.resize(n_coord);
    auto const it_coord_nn = v_coord_nn.begin();
    auto const it_coord_cell_size = v_coord_cell_size.begin();
    auto const it_nc_lvl_size = v_nc_lvl_size.begin();
    auto const it_nc_lvl_offset = v_nc_lvl_offset.begin();
    auto const it_coord_cell_index = v_coord_cell_index.begin();
    auto const it_coord_cell_offset = v_coord_cell_offset.begin();
    auto const it_coord = v_coord.begin();
    auto const it_cell_AABB = v_cell_AABB.begin();

    for (int c1 = 0; c1 < it_nc_lvl_size[0]; ++c1) {
        int cell_size = it_coord_cell_size[it_nc_lvl_offset[0] + c1];
        for (int i = 0; i < cell_size; ++i) {
            int p1 = it_coord_cell_index[it_nc_lvl_offset[0] + it_coord_cell_offset[c1] + i];
//            it_coord_nn[p1] = cell_size;
        }
    }

    // expand bounding boxes for reach

//    std::cout << v_cell_AABB.size() << " : " << v_coord_cell_size.size() << std::endl;
    for (std::size_t i = 0; i < v_coord_cell_size.size(); ++i) {
        int d = 0;
        for (; d < n_dim; ++d) {
            v_cell_AABB[(i * n_dim * 2) + d] -= e;
        }
        for (; d < 2 * n_dim; ++d) {
            v_cell_AABB[(i * n_dim * 2) + d] += e;
        }
    }

//    int track_height = static_cast<int>((v_nc_lvl_size.size() - 2));
    int track_height = 0;
    d_vec<int> v_tracker(n_coord * track_height * 2, 0);
    auto const it_tracker = v_tracker.begin();
    exa::for_each(0, 200, [=]
#ifdef CUDA_ON
    __device__
#endif
    (auto const &i) -> void {
        int hits = 0;
//        if (i % 10000 == 0)
//            std::cout << "i: " << i << std::endl;
        auto const p = &it_coord[i * n_dim];
        auto const stack = &it_tracker[i * track_height];
//        int lvl = track_height - 1;
        for (int c1 = 0; c1 < it_nc_lvl_size[track_height - 1]; ++c1) {
            int lvl = track_height - 1;
            stack[lvl * 2] = c1;
            stack[lvl * 2 + 1] = 0;
            do {
                int c = stack[lvl * 2];
                int j = stack[lvl * 2 + 1];
                bool are_connected = true;
                if (j == 0) {
                    auto const min2 = &it_cell_AABB[((it_nc_lvl_offset[lvl] + c) * n_dim * 2)];
                    auto const max2 = &it_cell_AABB[((it_nc_lvl_offset[lvl] + c) * n_dim * 2) + n_dim];
//                    if (lvl < 6) {
//                        magma_util::print_v("min: ", min2, n_dim);
//                        magma_util::print_v("max: ", max2, n_dim);
//                    }
                    for (int d = 0; d < n_dim; ++d) {
                        if (p[d] < min2[d] || p[d] > max2[d]) {
                            are_connected = false;
                            break;
                        }
                    }
                }
                if (are_connected) {
                    if (j < it_coord_cell_size[it_nc_lvl_offset[lvl] + c]) {
                        if (j > 0)
                            std::cout << "i: " << i << " c: " << c << " j: " << j << " lvl: " << lvl << std::endl;
                        stack[lvl * 2 + 1] = j + 1;
                        if (lvl == 0) {
                            for (int k = 0; k < it_coord_cell_size[c]; ++k) {
                                auto const p2 = &it_coord[it_coord_cell_index[it_coord_cell_offset[c] + k] * n_dim];
                                float length = 0;
                                for (int d = 0; d < n_dim; ++d) {
                                    length += (p[d] - p2[d]) * (p[d] - p2[d]);
                                    if (length > e2) {
                                        break;
                                    }
                                }
                                if (length <= e2) {
                                    if (++hits == m) {
                                        it_coord_nn[i] = m;
                                        return;
                                    }
                                }
                            }
                        } else {
                            --lvl;
                            stack[lvl * 2] = it_coord_cell_index[n_coord + it_nc_lvl_offset[lvl] + it_coord_cell_offset[it_nc_lvl_offset[lvl+1] + c] + j];
                            stack[lvl * 2 + 1] = 0;
                        }
                    } else {
                        ++lvl;
                    }
                } else {
//                    std::cout << "ELSE 3" << std::endl;
                    ++lvl;
                }
            } while (lvl < track_height - 1);
        }
//            for (int c = 0; c < it_nc_lvl_size[track_height - lvl]; ++c) {
//
//            }
//        }
//        auto first = i * (v_nc_lvl_size.size() - 1);
//        auto last = first + track_height - 1;
//        auto curr = first;
//        it_tracker[first] = -1;
//        while(it_tracker[first] < v_nc_lvl_size[track_height]) {
//            ++it_tracker[curr];
//        }

        /*
        for (int c = 0; c < it_nc_lvl_size[0]; ++c) {
            bool are_connected = true;
            auto const min2 = &it_cell_AABB[((it_nc_lvl_offset[0] + c) * n_dim * 2)];
            auto const max2 = &it_cell_AABB[((it_nc_lvl_offset[0] + c) * n_dim * 2) + n_dim];
            for (int d = 0; d < n_dim; ++d) {
                if (p[d] < min2[d] || p[d] > max2[d]) {
                    are_connected = false;
                    break;
                }
            }
            if (are_connected) {
                for (int j = 0; j < it_coord_cell_size[it_nc_lvl_offset[0] + c]; ++j) {
                    auto const p2 = &it_coord[it_coord_cell_index[it_nc_lvl_offset[0] + it_coord_cell_offset[c] + j] * n_dim];
                    float length = 0;
                    for (int d = 0; d < n_dim; ++d) {
                        length += (p[d] - p2[d]) * (p[d] - p2[d]);
                        if (length > e2) {
                            break;
                        }
                    }
                    if (length <= e2) {
                        if (++hits == m) {
                            it_coord_nn[i] = m;
                            return;
                        }
                    }
                }
            }
        }
        */
    });

    /*
    int track_height = static_cast<int>((v_nc_lvl_size.size() - 1));
    d_vec<int> v_tracker(n_coord * track_height, 0);
    auto const it_tracker = v_tracker.begin();
    for (std::size_t i = 0; i < n_coord; ++i) {
        auto const p = &it_coord[i * n_dim];
        int l = track_height;
        auto first = i * (v_nc_lvl_size.size() - 1);
        auto last = first + track_height - 1;
        auto curr = first;
        it_tracker[first] = -1;
        while(it_tracker[first] < v_nc_lvl_size[track_height]) {
            ++it_tracker[curr];
        }
    }
     */


    /*
    for (int c1 = 0; c1 < it_nc_lvl_size[0]; ++c1) {
        for (int c2 = c1 + 1; c2 < it_nc_lvl_size[0]; ++c2) {
            bool are_connected = true;
            auto const min1 = &it_cell_AABB[((it_nc_lvl_offset[0] + c1) * _n_dim * 2)];
            auto const max1 = &it_cell_AABB[((it_nc_lvl_offset[0] + c1) * _n_dim * 2) + _n_dim];
            auto const min2 = &it_cell_AABB[((it_nc_lvl_offset[0] + c2) * _n_dim * 2)];
            auto const max2 = &it_cell_AABB[((it_nc_lvl_offset[0] + c2) * _n_dim * 2) + _n_dim];

            for (int d = 0; d < _n_dim; ++d) {
                if ((min2[d] > (max1[d] + _e) || min2[d] < (min1[d] - _e)) &&
                    (min1[d] > (max2[d] + _e) || min1[d] < (min2[d] - _e)) &&
                    (max2[d] > (max1[d] + _e) || max2[d] < (min1[d] - _e)) &&
                    (max1[d] > (max2[d] + _e) || max1[d] < (min2[d] - _e))) {
                    are_connected = false;
                    break;
                }
            }
            if (are_connected) {
                for (int i = 0; i < it_coord_cell_size[it_nc_lvl_offset[0] + c1]; ++i) {
                    int p1 = it_coord_cell_index[it_nc_lvl_offset[0] + it_coord_cell_offset[c1] + i];
                    for (int j = 0; j < it_coord_cell_size[it_nc_lvl_offset[0] + c2]; ++j) {
                        int p2 = it_coord_cell_index[it_nc_lvl_offset[0] + it_coord_cell_offset[c2] + j];
                        float length = 0;
                        for (int d = 0; d < _n_dim; ++d) {
                            length += (it_coord[p1 * _n_dim + d] - it_coord[p2 * _n_dim + d]) *
                                      (it_coord[p1 * _n_dim + d] - it_coord[p2 * _n_dim + d]);
                            if (length > _e2) {
                                break;
                            }
                        }
                        if (length <= _e2) {
                            it_coord_nn[p1] = it_coord_nn[p1] + 1;
                            it_coord_nn[p2] = it_coord_nn[p2] + 1;
                        }
                    }
                }
            }
        }
    }
     */
}

