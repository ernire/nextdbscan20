//
// Created by Ernir Erlingsson on 19.8.2020.
//

#include <iostream>
#include <unordered_map>
#include <limits>
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

void data_process::process_points2(d_vec<int> const &v_point_id, d_vec<float> const &v_point_data,
        d_vec<long long> &v_point_index, d_vec<int> &v_point_cells_in_reach,
        d_vec<int> &v_point_cell_reach_offset, d_vec<int> &v_point_cell_reach_size,
        magmaMPI mpi) noexcept {
    // TODO move this to the parent

    // calculate cell index
    index_points(v_point_data, v_point_index);
    // obtain reach
    collect_cells_in_reach(v_point_index, v_point_cells_in_reach, v_point_cell_reach_offset,
            v_point_cell_reach_size);


    // count the total number of possible hits for the cells

    // eliminate all points that belong to cells that can't contain cores

    // count eps / sqrt(2) bounding box

    //
    d_vec<int> v_point_nn(v_point_id.size(), 0);
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
    auto const it_point_id = v_point_id.begin();
    auto const _n_dim = n_dim;
    auto const _e2 = e2;
    auto const _e_inner = (e / sqrtf(2)) / 2;
    auto const _m = m;

    // Finna cores
    exa::for_each(0, v_point_id.size(), [=]
#ifdef CUDA_ON
            __device__
#endif
    (auto const &i) -> void {
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
    d_vec<int> v_point_iota(v_point_id.size());
    d_vec<int> v_point_core_id(v_point_id.size());
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
//    std::cout << "cores/chunk ratio = " << v_point_core_id.size() << " / " << v_point_id.size() << std::endl;

    // setja nýtt core cluster label ef þörf er á
    auto const _cluster_size = v_cluster_label.size();
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
                    if (it_cluster_label[it_coord_cluster_index[id2]] != it_point_cluster_other[k]/*it_cluster_label[it_point_cluster_index[k]]*/) {
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
            // merge
            do {
                v_running2[0] = 0;
                exa::for_each(0, v_point_core_id.size(), [=]
#ifdef CUDA_ON
                __device__
#endif
                (auto const &k) -> void {
                    if (it_cluster_label[it_point_cluster_index[k]] < it_cluster_label[it_point_cluster_other[k]]) {
                        exa::_atomic_op(&it_cluster_label[it_point_cluster_other[k]],
                                it_cluster_label[it_point_cluster_index[k]], std::less<>());
                        it_running2[0] = 1;
                    } else if (it_cluster_label[it_point_cluster_index[k]] >
                               it_cluster_label[it_point_cluster_other[k]]) {
                        exa::_atomic_op(&it_cluster_label[it_point_cluster_index[k]],
                                it_cluster_label[it_point_cluster_other[k]], std::less<>());
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

    return;

    exa::for_each(0, v_point_id.size(), [=]
#ifdef CUDA_ON
            __device__
#endif
    (auto const &i) -> void {
        // check if this point has already been classified as a core
        if (it_point_id[i] >= 0 && it_coord_nn[it_point_id[i]] >= _m) {
//            #pragma omp critical
//            std::cout << "SKIPPED!" << std::endl;
            return;
        }
        int outer_nn = 0;
        int inner_nn = 0;
        bool is_inner;
        float length;
        for (int ci = 0; ci < it_point_cell_reach_size[i]; ++ci) {
            int const c = it_point_cells_in_reach[it_point_cell_reach_offset[i] + ci];
            for (int j = 0; j < it_coord_cell_size[c]; ++j) {
                int const id2 = it_coord_id[it_coord_cell_offset[c] + j];
                is_inner = true;
                // check inner reach
                for (int d = 0; d < _n_dim; ++d) {
                    if (it_point_data[i * _n_dim + d] - it_coord_data[id2 * _n_dim + d] < -_e_inner
                        || it_point_data[i * _n_dim + d] - it_coord_data[id2 * _n_dim + d] > _e_inner) {
                        is_inner = false;
                        break;
                    }
                }
                if (is_inner) {
                    ++inner_nn;
                    continue;
                }
                // check outer reach only if necessary
                if (outer_nn + inner_nn < _m) {
                    length = 0;
                    for (int d = 0; d < _n_dim; ++d) {
                        length += (it_point_data[i * _n_dim + d] - it_coord_data[id2 * _n_dim + d]) *
                                  (it_point_data[i * _n_dim + d] - it_coord_data[id2 * _n_dim + d]);
                    }
                    if (length <= _e2) {
                        ++outer_nn;
                    }
                }
            }
        }
        it_point_nn[i] = inner_nn + outer_nn;
        if (it_point_id[i] >= 0) {
            it_coord_nn[it_point_id[i]] = it_point_nn[i];
        }
        /*
        if (inner_nn >= _m) {
//            #pragma omp critical
//            std::cout << "INNER 1!" << std::endl;
            for (int ci = 0; ci < it_point_cell_reach_size[i]; ++ci) {
                int const c = it_point_cells_in_reach[it_point_cell_reach_offset[i] + ci];
                for (int j = 0; j < it_coord_cell_size[c]; ++j) {
                    int const id2 = it_coord_id[it_coord_cell_offset[c] + j];
                    if (it_coord_nn[id2] >= _m)
                        continue;
                    is_inner = true;
                    // check inner reach
                    for (int d = 0; d < _n_dim; ++d) {
                        if (it_point_data[i * _n_dim + d] - it_coord_data[id2 * _n_dim + d] < -_e_inner
                            || it_point_data[i * _n_dim + d] - it_coord_data[id2 * _n_dim + d] > _e_inner) {
                            is_inner = false;
                            break;
                        }
                    }
                    if (is_inner) {
//                        #pragma omp critical
//                        std::cout << "INNER 2! " << inner_nn << std::endl;
                        it_coord_nn[id2] = inner_nn;
                    }
                }
            }
        }
         */
    });
//    // Determine the cluster label (needs to be checked among all)
//    auto const _cluster_size = v_cluster_label.size();
//    d_vec<int> v_point_cluster_index(v_point_id.size(), INT32_MAX);
//    auto const it_point_cluster_index = v_point_cluster_index.begin();
//    auto const it_coord_cluster_index = v_coord_cluster_index.begin();
//    d_vec<int> v_point_new_cluster_mark(v_point_id.size(), 0);
//    d_vec<int> v_point_new_cluster_offset(v_point_id.size(), 0);
//    auto const it_point_new_cluster_mark = v_point_new_cluster_mark.begin();
    // Initialize the point cluster label
    exa::for_each(0, v_point_id.size(), [=]
#ifdef CUDA_ON
            __device__
#endif
        (auto const &i) -> void {
//        int const i = it_sorted_order[k];
        if (it_point_nn[i] < _m) return;
        else if (it_point_id[i] >= 0) {
            // Discover which new cores have not been labeled
            if (it_coord_cluster_index[it_point_id[i]] == NO_CLUSTER) {
                // mark for a new label
                it_point_new_cluster_mark[i] = 1;
            } else {
                it_point_cluster_index[i] = it_coord_cluster_index[it_point_id[i]];
            }

        }
    });
    // count the new labels
    /*
    auto new_cluster_cores = exa::count_if(v_point_new_cluster_mark, 0, v_point_new_cluster_mark.size(), []
#ifdef CUDA_ON
            __device__
#endif
    (auto const &v) -> bool {
        return v == 1;
    });

#ifdef DEBUG_ON
    std::cout << "new cluster cores: " << new_cluster_cores << std::endl;
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
    exa::for_each(0, v_point_id.size(), [=]
#ifdef CUDA_ON
            __device__
#endif
    (auto const &i) -> void {
        if (it_point_nn[i] < _m) return;
        else if (it_point_new_cluster_mark[i] == 1) {
            // mark the new cluster indexes
            it_point_cluster_index[i] = it_point_new_cluster_offset[i];
            if (it_point_id[i] >= 0) {
                it_coord_cluster_index[it_point_id[i]] = it_point_cluster_index[i];
            }
        }
    });
    auto const it_cluster_label = v_cluster_label.begin();
    d_vec<int> v_running(1);
    auto const it_running = v_running.begin();
    int iter_cnt = 0;

    d_vec<int> v_point_cluster_minmax(v_point_id.size() * 2, -1);
    auto const it_point_cluster_minmax = v_point_cluster_minmax.begin();
    // sort out the non-used points
    do {
        ++iter_cnt;
        v_running[0] = 0;
        exa::for_each(0, v_point_id.size(), [=]
#ifdef CUDA_ON
        __device__
#endif
        (auto const &i) -> void {
            if (it_point_nn[i] < _m) return;
            // min
            it_point_cluster_minmax[i*2] = it_cluster_label[it_point_cluster_index[i]];
            // max
            it_point_cluster_minmax[i*2+1] = it_cluster_label[it_point_cluster_index[i]];
            for (int ci = 0; ci < it_point_cell_reach_size[i]; ++ci) {
                int const c = it_point_cells_in_reach[it_point_cell_reach_offset[i] + ci];
                for (int j = 0; j < it_coord_cell_size[c]; ++j) {
                    int const id2 = it_coord_id[it_coord_cell_offset[c] + j];
                    if (it_coord_nn[id2] < _m) continue;
                    if (it_cluster_label[it_coord_cluster_index[id2]] < it_point_cluster_minmax[i*2]) {
                        if (dist_leq(&it_point_data[i * _n_dim], &it_coord_data[id2 * _n_dim], _n_dim, _e2)) {
                            it_point_cluster_minmax[i * 2] = it_cluster_label[it_coord_cluster_index[id2]];
                        }
//                        if (it_cluster_label[it_coord_cluster_index[id2]] > it_point_cluster_minmax[i*2+1])
//                            std::cerr << "ERRROR!!!" << std::endl;
                    } else if (it_cluster_label[it_coord_cluster_index[id2]] > it_point_cluster_minmax[i*2+1]) {
                        if (dist_leq(&it_point_data[i * _n_dim], &it_coord_data[id2 * _n_dim], _n_dim, _e2)) {
                            it_point_cluster_minmax[i * 2 + 1] = it_cluster_label[it_coord_cluster_index[id2]];
                        }
//                        if (it_cluster_label[it_coord_cluster_index[id2]] < it_point_cluster_minmax[i*2])
//                            std::cerr << "ERRROR!!!" << std::endl;
                    }
                }
            }

        });


    } while (v_running[0] > 0);
          */
    /*
    // Determine core labels
    do {
        ++iter_cnt;
        v_running[0] = 0;
        exa::for_each(0, v_point_id.size(), [=]
#ifdef CUDA_ON
            __device__
#endif
        (auto const &i) -> void {
            if (it_point_nn[i] < _m) return;

            assert(it_point_cluster_index[i] >= 0 && it_point_cluster_index[i] < _cluster_label_size);
            for (int ci = 0; ci < it_point_cell_reach_size[i]; ++ci) {
                int const c = it_point_cells_in_reach[it_point_cell_reach_offset[i] + ci];
                for (int j = 0; j < it_coord_cell_size[c]; ++j) {
                    int const id2 = it_coord_id[it_coord_cell_offset[c] + j];
                    if (it_coord_nn[id2] < _m) continue;
                    if (it_cluster_label[it_point_cluster_index[i]] < it_cluster_label[it_coord_cluster_index[id2]]) {
                        if (dist_leq(&it_point_data[i * _n_dim], &it_coord_data[id2 * _n_dim], _n_dim, _e2)) {
                            exa::_atomic_op(&it_cluster_label[it_coord_cluster_index[id2]],
                                    it_cluster_label[it_point_cluster_index[i]], std::less<>());
                            it_running[0] = 1;
                        }
                    } else if (it_cluster_label[it_point_cluster_index[i]] > it_cluster_label[it_coord_cluster_index[id2]]) {
                        if (dist_leq(&it_point_data[i * _n_dim], &it_coord_data[id2 * _n_dim], _n_dim, _e2)) {
                            exa::_atomic_op(&it_cluster_label[it_point_cluster_index[i]],
                                    it_cluster_label[it_coord_cluster_index[id2]], std::less<>());
                            it_running[0] = 1;
                        }
                    }
                }
            }
        });

        // flatten
        exa::for_each(0, v_coord_id.size(), [=]
#ifdef CUDA_ON
                __device__
#endif
                (auto const &k) -> void {
            int const i = it_coord_id[k];
            if (it_coord_cluster_index[i] == NO_CLUSTER)
                return;
            while (it_cluster_label[it_coord_cluster_index[i]] != it_coord_cluster_index[i]) {
                it_coord_cluster_index[i] = it_cluster_label[it_coord_cluster_index[i]];
            }
        });
    } while (v_running[0] > 0);
    */
//#ifdef DEBUG_ON
//    std::cout << "Label Iterations: " << iter_cnt << std::endl;
//#endif

/*
    // Label non-cores
    exa::for_each(0, v_point_id.size(), [=]
#ifdef CUDA_ON
            __device__
#endif
    (auto const &i) -> void {
        if (it_point_nn[i] < _m) return;
        for (int ci = 0; ci < it_point_cell_reach_size[i]; ++ci) {
            int const c = it_point_cells_in_reach[it_point_cell_reach_offset[i] + ci];
            for (int j = 0; j < it_coord_cell_size[c]; ++j) {
                int const id2 = it_coord_id[it_coord_cell_offset[c] + j];
                if (it_coord_nn[id2] >= _m) continue;
                if (it_coord_cluster_index[id2] == NO_CLUSTER) {
                    if (dist_leq(&it_point_data[i * _n_dim], &it_coord_data[id2 * _n_dim], _n_dim, _e2)) {
                        exa::_atomic_op(&it_coord_cluster_index[id2], it_cluster_label[it_point_cluster_index[i]],
                                std::less<>());
                    }
                }
                /*
                else if (it_cluster_label[it_point_cluster_index[i]] < it_cluster_label[it_coord_cluster_index[id2]]) {
                    if (dist_leq(&it_point_data[i * _n_dim], &it_coord_data[id2 * _n_dim], _n_dim, _e2)) {
                        exa::_atomic_op(&it_coord_cluster_index[id2],
                                it_cluster_label[it_point_cluster_index[i]], std::less<>());
                        if (it_coord_status[id2] == NOT_PROCESSED) {
                            it_coord_status[id2] = MARKED;
                        }
                    }
                } else if (it_cluster_label[it_point_cluster_index[i]] > it_cluster_label[it_coord_cluster_index[id2]]
                           && it_coord_status[id2] == NOT_PROCESSED) {
                    if (dist_leq(&it_point_data[i * _n_dim], &it_coord_data[id2 * _n_dim], _n_dim, _e2)) {
                        it_coord_status[id2] = MARKED;
                    }
                }
                */
//            }
//        }
//    });
    /*
    // flatten again
    exa::for_each(0, v_coord_id.size(), [=]
#ifdef CUDA_ON
            __device__
#endif
            (auto const &k) -> void {
        int const i = it_coord_id[k];
        if (it_coord_cluster_index[i] == NO_CLUSTER)
            return;
        while (it_cluster_label[it_coord_cluster_index[i]] != it_coord_cluster_index[i]) {
            it_coord_cluster_index[i] = it_cluster_label[it_coord_cluster_index[i]];
        }
    });
*/
    return;




    if (v_coord_cell_id.empty()) {
        // Store the coord to cell map
        v_coord_cell_id.resize(n_coord);
        auto const it_coord_cell_id = v_coord_cell_id.begin();
        exa::for_each(0, v_coord_cell_size.size(), [=]
#ifdef CUDA_ON
        __device__
#endif
        (auto const &i) -> void {
            for (int j = 0; j < it_coord_cell_size[i]; ++j) {
                int const id2 = it_coord_id[it_coord_cell_offset[i] + j];
                it_coord_cell_id[id2] = i;
            }
        });
    }
    auto const it_coord_cell_id = v_coord_cell_id.begin();
//    auto const it_point_id = v_point_id.begin();
    d_vec<uint64_t> v_cell_use(v_coord_cell_size.size());
    d_vec<uint64_t> v_hit(n_coord);
//    d_vec<int> v_point_nn(v_point_id.size(), 0);
    auto const it_cell_use = v_cell_use.begin();
    auto const it_hit = v_hit.begin();
//    auto const it_point_data = v_point_data.begin();
//    auto const it_coord_data = v_coord.begin();
//    auto const it_point_nn = v_point_nn.begin();
//    auto const it_coord_nn = v_coord_nn.begin();
//    auto const _n_dim = n_dim;
//    auto const _e2 = e2;
//    auto const _m = m;
    std::size_t point_begin = 0;

    do {
        std::size_t point_end = point_begin + 64;
        if (point_end >= v_point_id.size()) {
            point_end = v_point_id.size();
        }
        exa::fill(v_cell_use, 0, v_cell_use.size(), static_cast<uint64_t>(0));
        exa::fill(v_hit, 0, v_hit.size(), static_cast<uint64_t>(0));

        exa::for_each(0, point_end - point_begin, [=]
#ifdef CUDA_ON
        __device__
#endif
        (auto const &i) -> void {
            for (int ci = 0; ci < it_point_cell_reach_size[i + point_begin]; ++ci) {
                int const c = it_point_cells_in_reach[it_point_cell_reach_offset[i + point_begin] + ci];
                // TODO
//                #pragma omp atomic
//                it_cell_use[c] += it_bitmask64[i];
//                it_cell_use[c] = it_cell_use[c] + it_bitmask64[i];
//                it_cell_use[c] += static_cast<uint64_t>(1) << i;
                exa::atomic_add(it_cell_use[c], static_cast<uint64_t>(1) << i);
            }
        });
        // determine the hit masks
        exa::for_each(0, n_coord, [=]
#ifdef CUDA_ON
        __device__
#endif
        (auto const &i) -> void {
            int hit_mask = 0;
            for (int j = 0; j < point_end - point_begin; ++j) {
                if ((it_cell_use[it_coord_cell_id[i]] & (static_cast<uint64_t>(1) << j)) > 0) {
//                if ((it_cell_use[it_coord_cell_id[i]] & it_bitmask64[j]) > 0) {
                    float length = 0;
                    for (int d = 0; d < _n_dim; ++d) {
                        length += (it_point_data[(point_begin + j) * _n_dim + d] - it_coord_data[i * _n_dim + d]) *
                                (it_point_data[(point_begin + j) * _n_dim + d] - it_coord_data[i * _n_dim + d]);
                    }
                    if (length <= _e2) {
//                        hit_mask += it_bitmask64[j];
                        hit_mask += (static_cast<uint64_t>(1) << j);
                        if (it_point_nn[point_begin + j] < _m) {
                            // TODO generalize
//                            #pragma omp atomic
//                            it_point_nn[point_begin + j] = it_point_nn;
                            exa::atomic_add(it_point_nn[point_begin + j], 1);
                        }
                    }


                    /*
                    if (dist_leq(&it_point_data[(point_begin + j) * _n_dim], &it_coord_data[i * _n_dim],
                            _n_dim, _e2)) {

//                        hit_mask += bitmask64[j];
                        hit_mask += (static_cast<uint64_t>(1) << j);
                        if (it_point_nn[point_begin + j] < _m) {
                            // TODO generalize
                            #pragma omp atomic
                            ++it_point_nn[point_begin + j];
                        }
                    }
                     */
                }
            }
            if (hit_mask > 0) {
                it_hit[i] = hit_mask;
            }
        });
        exa::for_each(0, point_end - point_begin, [=]
#ifdef CUDA_ON
            __device__
#endif
        (auto const &i) -> void {
            if (it_point_nn[point_begin + i] >= _m && it_point_id[point_begin + i] >= 0) {
                it_coord_nn[it_point_id[point_begin + i]] = _m;
            }
        });
        point_begin += 64;
    } while(point_begin < v_point_id.size());





}

void data_process::process_points(d_vec<int> &v_point_id, d_vec<float> &v_point_data, magmaMPI mpi) noexcept {
    auto const it_coord_status = v_coord_status.begin();
    auto const it_point_id = v_point_id.begin();

    // calculate cell index
    d_vec<long long> v_point_index(v_point_id.size());
    index_points(v_point_data, v_point_index);

    // obtain reach
    d_vec<int> v_point_cells_in_reach(v_point_id.size());
    d_vec<int> v_point_cell_reach_offset(v_point_id.size());
    d_vec<int> v_point_cell_reach_size(v_point_id.size());

    collect_cells_in_reach(v_point_index, v_point_cells_in_reach, v_point_cell_reach_offset,
            v_point_cell_reach_size);

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
    (auto const &i) -> void {
        auto p_sum = 0;
        for (auto j = 0; j < it_point_cell_reach_size[i]; ++j) {
            p_sum += it_coord_cell_size[it_point_cells_in_reach[it_point_cell_reach_offset[i] + j]];
        }
        it_points_in_reach_size[i] = p_sum;
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
    });

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
    (auto const &k) -> void {
        int const i = it_sorted_order[k];
        if (it_point_id[i] >= 0) {
            it_point_nn[i] = it_coord_nn[it_point_id[i]];
            it_coord_status[it_point_id[i]] = MARKED;
            it_coord_nn[it_point_id[i]] = 0;
        }
    });

    exa::for_each(0, v_point_id.size(), [=]
#ifdef CUDA_ON
    __device__
#endif
    (auto const &k) -> void {
        int const i = it_sorted_order[k];
        int nn = it_point_nn[i];
        for (int ci = 0; ci < it_point_cell_reach_size[i]; ++ci) {
            int const c = it_point_cells_in_reach[it_point_cell_reach_offset[i] + ci];
            /*
            for (int j = 0; j < it_coord_cell_size[c]; ++j) {
                int const id2 = it_coord_id[it_coord_cell_offset[c] + j];
                if (it_coord_status[id2] == PROCESSED || (nn >= _m && it_coord_nn[id2] >= _m)) {
                    continue;
                } else if (it_point_id[i] >= 0 && it_coord_status[it_point_id[i]] == MARKED
                    && it_coord_status[id2] == MARKED && it_point_id[i] >= id2) {
                        continue;
                }
                // continue if both are already cores
                if (dist_leq(&it_point_data[i * _n_dim], &it_coord_data[id2 * _n_dim], _n_dim, _e2)) {
                    ++nn;
                    #pragma omp atomic
                    ++it_coord_nn[id2];
                }
            }
             */
        }
        it_point_nn[i] = nn;
        /*
        if (it_point_id[i] >= 0) {
            it_coord_nn[it_point_id[i]] += it_point_nn[i];
            it_point_nn[i] = it_coord_nn[it_point_id[i]];
        }
         */
    });


    /*
#ifdef MPI_ON
    mpi.allReduce(v_point_nn, magmaMPI::max);
#endif

    d_vec<int> v_point_new_cluster_mark(v_point_id.size(), 0);
    d_vec<int> v_point_new_cluster_offset(v_point_id.size(), 0);
    auto const it_point_new_cluster_mark = v_point_new_cluster_mark.begin();
    d_vec<int> v_point_cluster_index(v_point_id.size(), INT32_MAX);
    auto const it_point_cluster_index = v_point_cluster_index.begin();
    auto const it_coord_cluster_index = v_coord_cluster_index.begin();
    auto const _cluster_size = v_cluster_label.size();

    // Initialize the point cluster label
    exa::for_each(0, v_point_id.size(), [=]
#ifdef CUDA_ON
            __device__
#endif
    (auto const &k) -> void {
        int const i = it_sorted_order[k];
        if (it_point_nn[i] < _m) return;
        else if (it_point_id[i] >= 0) {
            // Discover which new cores have not been labeled
            if (it_coord_cluster_index[it_point_id[i]] == NO_CLUSTER) {
                // mark for a new label
                it_point_new_cluster_mark[i] = 1;
            } else {
                it_point_cluster_index[i] = it_coord_cluster_index[it_point_id[i]];
            }

        }
    });
#ifdef MPI_ON
    mpi.allReduce(v_point_new_cluster, magmaMPI::max);
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
    std::cout << "new cluster cores: " << new_cluster_cores << std::endl;
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
    exa::for_each(0, v_point_id.size(), [=]
#ifdef CUDA_ON
        __device__
#endif
    (auto const &k) -> void {
        int const i = it_sorted_order[k];
        if (it_point_nn[i] < _m) return;
        else if (it_point_new_cluster_mark[i] == 1) {
            // mark the new cluster indexes
            it_point_cluster_index[i] = it_point_new_cluster_offset[i];
            if (it_point_id[i] >= 0) {
                it_coord_cluster_index[it_point_id[i]] = it_point_cluster_index[i];
            }
        }
    });
#ifdef MPI_ON
    mpi.allReduce(v_point_new_cluster, magmaMPI::max);
#endif

    auto const it_cluster_label = v_cluster_label.begin();
    d_vec<int> v_running(1);
    auto const it_running = v_running.begin();
    int iter_cnt = 0;

    // Determine core labels
    do {
        ++iter_cnt;
        v_running[0] = 0;
        exa::for_each(0, v_point_id.size(), [=]
#ifdef CUDA_ON
        __device__
#endif
        (auto const &k) -> void {
            int const i = it_sorted_order[k];
            if (it_point_nn[i] < _m) return;

            assert(it_point_cluster_index[i] >= 0 && it_point_cluster_index[i] < _cluster_label_size);
            for (int ci = 0; ci < it_point_cell_reach_size[i]; ++ci) {
                int const c = it_point_cells_in_reach[it_point_cell_reach_offset[i] + ci];
                for (int j = 0; j < it_coord_cell_size[c]; ++j) {
                    int const id2 = it_coord_id[it_coord_cell_offset[c] + j];
                    if (it_coord_nn[id2] < _m) continue;
                    if (it_cluster_label[it_point_cluster_index[i]] < it_cluster_label[it_coord_cluster_index[id2]]) {
                        if (dist_leq(&it_point_data[i * _n_dim], &it_coord_data[id2 * _n_dim], _n_dim, _e2)) {
                            exa::_atomic_op(&it_cluster_label[it_coord_cluster_index[id2]],
                                    it_cluster_label[it_point_cluster_index[i]], std::less<>());
                            it_running[0] = 1;
                        }
                    } else if (it_cluster_label[it_point_cluster_index[i]] > it_cluster_label[it_coord_cluster_index[id2]]) {
                        if (dist_leq(&it_point_data[i * _n_dim], &it_coord_data[id2 * _n_dim], _n_dim, _e2)) {
                            exa::_atomic_op(&it_cluster_label[it_point_cluster_index[i]],
                                    it_cluster_label[it_coord_cluster_index[id2]], std::less<>());
                            it_running[0] = 1;
                        }
                    }
                }
            }
        });

        // flatten
        exa::for_each(0, v_coord_id.size(), [=]
#ifdef CUDA_ON
        __device__
#endif
    (auto const &k) -> void {
            int const i = it_coord_id[k];
            if (it_coord_cluster_index[i] == NO_CLUSTER)
                return;
            while (it_cluster_label[it_coord_cluster_index[i]] != it_coord_cluster_index[i]) {
                it_coord_cluster_index[i] = it_cluster_label[it_coord_cluster_index[i]];
            }
        });
    } while (v_running[0] > 0);

#ifdef DEBUG_ON
    std::cout << "Label Iterations: " << iter_cnt << std::endl;
#endif

    // Label non-cores
    exa::for_each(0, v_point_id.size(), [=]
#ifdef CUDA_ON
        __device__
#endif
    (auto const &k) -> void {
        int const i = it_sorted_order[k];
        if (it_point_nn[i] < _m) return;
        for (int ci = 0; ci < it_point_cell_reach_size[i]; ++ci) {
            int const c = it_point_cells_in_reach[it_point_cell_reach_offset[i] + ci];
            for (int j = 0; j < it_coord_cell_size[c]; ++j) {
                int const id2 = it_coord_id[it_coord_cell_offset[c] + j];
                if (it_coord_nn[id2] >= _m) continue;
                if (it_coord_cluster_index[id2] == NO_CLUSTER) {
                    if (dist_leq(&it_point_data[i * _n_dim], &it_coord_data[id2 * _n_dim], _n_dim, _e2)) {
                        exa::_atomic_op(&it_coord_cluster_index[id2], it_cluster_label[it_point_cluster_index[i]],
                                std::less<>());
                    }
                } else if (it_cluster_label[it_point_cluster_index[i]] < it_cluster_label[it_coord_cluster_index[id2]]) {
                    if (dist_leq(&it_point_data[i * _n_dim], &it_coord_data[id2 * _n_dim], _n_dim, _e2)) {
                        exa::_atomic_op(&it_coord_cluster_index[id2],
                                it_cluster_label[it_point_cluster_index[i]], std::less<>());
                        if (it_coord_status[id2] == NOT_PROCESSED) {
                            it_coord_status[id2] = MARKED;
                        }
                    }
                } else if (it_cluster_label[it_point_cluster_index[i]] > it_cluster_label[it_coord_cluster_index[id2]]
                    && it_coord_status[id2] == NOT_PROCESSED) {
                    if (dist_leq(&it_point_data[i * _n_dim], &it_coord_data[id2 * _n_dim], _n_dim, _e2)) {
                        it_coord_status[id2] = MARKED;
                    }
                }
            }
        }
    });

    // flatten again
    exa::for_each(0, v_coord_id.size(), [=]
#ifdef CUDA_ON
        __device__
#endif
    (auto const &k) -> void {
        int const i = it_coord_id[k];
        if (it_coord_cluster_index[i] == NO_CLUSTER)
            return;
        while (it_cluster_label[it_coord_cluster_index[i]] != it_coord_cluster_index[i]) {
            it_coord_cluster_index[i] = it_cluster_label[it_coord_cluster_index[i]];
        }
    });
//#ifdef CUDA_ON
//    print_cuda_memory_usage();
//#endif
    */
    exa::for_each(0, v_point_id.size(), [=]
#ifdef CUDA_ON
            __device__
#endif
    (int const &i) -> void {
        if (it_point_id[i] >= 0)
            it_coord_status[it_point_id[i]] = PROCESSED;
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
    v_coord_cluster_index.resize(n_coord, NO_CLUSTER);
    v_coord_status.resize(n_coord, NOT_PROCESSED);
//    v_bitmask64.resize(64);
//    auto const it_bitmask64 = v_bitmask64.begin();
//    exa::for_each(0, 64, [=]
//#ifdef CUDA_ON
//    __device__
//#endif
//    (auto const &i) -> void {
//        it_bitmask64[i] = static_cast<uint64_t>(1) << i;
//    });


    auto const it_coord_status = v_coord_status.begin();
    int coord_begin = 0;
    int chunk_size = 32768;
//    int chunk_size = n_coord / 2;
    d_vec<int> v_point_id(chunk_size);
    d_vec<float> v_point_data(chunk_size * n_dim);
    d_vec<long long> v_point_index(v_point_id.size());
    d_vec<int> v_point_cells_in_reach(v_point_id.size());
    d_vec<int> v_point_cell_reach_offset(v_point_id.size());
    d_vec<int> v_point_cell_reach_size(v_point_id.size());
    do {
        exa::iota(v_point_id, 0, v_point_id.size(), coord_begin);
        int coord_end = coord_begin + chunk_size;
        if (coord_end > n_coord) {
            coord_end = n_coord;
            v_point_id.resize(coord_end - coord_begin);
            v_point_index.resize(v_point_id.size());
            v_point_cells_in_reach.resize(v_point_id.size());
            v_point_cell_reach_offset.resize(v_point_id.size());
            v_point_cell_reach_size.resize(v_point_id.size());
            v_point_data.resize(v_point_id.size() * n_dim);
        }
#ifdef CUDA_ON
        thrust::copy(v_coord.begin() + coord_begin * n_dim,
                v_coord.begin() + coord_end*n_dim,
                v_point_data.begin());
#else
        std::copy(std::next(v_coord.begin(), coord_begin * n_dim),
                std::next(v_coord.begin(), (coord_end)*n_dim),
                v_point_data.begin());
#endif
        process_points2(v_point_id, v_point_data, v_point_index, v_point_cells_in_reach,
                v_point_cell_reach_offset, v_point_cell_reach_size, mpi);
        coord_begin += chunk_size;
    } while (coord_begin < n_coord);
}

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
