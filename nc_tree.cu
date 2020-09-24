//
// Created by Ernir Erlingsson on 19.8.2020.
//

#include <iostream>
#include <unordered_map>
#include <thrust/extrema.h>
#include <thrust/pair.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/functional.h>
#include <thrust/binary_search.h>
#include "magma_util.h"
#include "nc_tree.h"

struct pack {
    template<typename Tuple>
    __device__ __host__ int64_t operator()(const Tuple &t) {
        return( static_cast<int64_t>( thrust::get<0>(t) ) << 32 ) | thrust::get<1>(t);
    }
};

struct unpack {
    __device__ __host__  thrust::tuple<int,int> operator()(int64_t p) {
        int32_t d = static_cast<int32_t>(p >> 32);
        int32_t s = static_cast<int32_t>(p & 0xffffffff);
        return thrust::make_tuple(d, s);
    }
};

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

void nc_tree::determine_data_bounds() noexcept {
    v_coord_id.resize(n_coord);
    thrust::sequence(v_coord_id.begin(), v_coord_id.end(), 0);
    v_min_bounds.resize(n_dim);
    v_max_bounds.resize(n_dim);
    thrust::counting_iterator<int> it_cnt_begin(0);
    thrust::counting_iterator<int> it_cnt_end = it_cnt_begin + n_coord;
    auto const i_begin = v_device_coord.begin();
    for (int d = 0; d < n_dim; ++d) {
        auto it_trans_begin = thrust::make_transform_iterator(it_cnt_begin, (thrust::placeholders::_1 * n_dim) + d);
        auto it_trans_end = thrust::make_transform_iterator(it_cnt_end, (thrust::placeholders::_1 * n_dim) + d);
        auto it_perm_begin = thrust::make_permutation_iterator(v_device_coord.begin(), it_trans_begin);
        auto it_perm_end = thrust::make_permutation_iterator(v_device_coord.end(), it_trans_end);
        auto minmax = thrust::minmax_element(it_perm_begin, it_perm_end);
        v_min_bounds[d] = *minmax.first;
        v_max_bounds[d] = *minmax.second;
    }
    v_dim_order.resize(n_dim);
    thrust::sequence(v_dim_order.begin(), v_dim_order.end(), 0);
    auto const i_min_begin = v_min_bounds.begin();
    auto const i_max_begin = v_max_bounds.begin();
    thrust::sort(v_dim_order.begin(), v_dim_order.end(), [=]__device__(int const &d1, int const &d2) -> bool {
        return (*(i_max_begin+d1) - *(i_min_begin+d1)) > (*(i_max_begin+d2) - *(i_min_begin+d2));
    });
}

void nc_tree::index_points(d_vec<float> &v_data, d_vec<int> &v_index) noexcept {
    thrust::counting_iterator<int> it_cnt_begin(0);
    thrust::counting_iterator<int> it_cnt_end = it_cnt_begin + v_index.size();
    auto const it_coords = v_data.begin();
    auto const dim_0 = v_dim_order[0];
    auto const dim_1 = v_dim_order[1];
    auto const bound_0 = v_min_bounds[dim_0];
    auto const bound_1 = v_min_bounds[dim_0];
    auto const mult = v_dim_part_size[0];
    int const dim = n_dim;
    float const ee = e;
    thrust::transform(it_cnt_begin, it_cnt_end, v_index.begin(), [=]__device__(int const &i) -> int {
        return (int)( ( *(it_coords + (i * dim + dim_0)) - bound_0 ) / ee )
               + (int)( ( *(it_coords + (i * dim + dim_1)) - bound_1 ) / ee ) * mult;
    });
}

void nc_tree::initialize_cells() noexcept {
    v_dim_part_size.resize(2);
    v_dim_part_size[0] = (v_max_bounds[v_dim_order[0]] - v_min_bounds[v_dim_order[0]]) / e + 1;
    v_dim_part_size[1] = (v_max_bounds[v_dim_order[1]] - v_min_bounds[v_dim_order[1]]) / e + 1;
//    magma_util::print_v("v_dim_part_size: ", &v_dim_part_size[0], v_dim_part_size.size());
    if (static_cast<uint64_t>(v_dim_part_size[0]) * v_dim_part_size[1] > INT32_MAX) {
        std::cerr << "FAIL: The epsilon value is too low and therefore not supported by the current version for the"
                     " input dataset" << std::endl;
        exit(EXIT_FAILURE);
    }
    v_coord_cell_index.resize(v_coord_id.size());
    d_vec<int> v_point_cell_index(v_coord_id.size());
    index_points(v_device_coord, v_point_cell_index);

    thrust::sort_by_key(v_point_cell_index.begin(), v_point_cell_index.end(), v_coord_id.begin());
    thrust::counting_iterator<int> it_cnt_begin(0);
    thrust::counting_iterator<int> it_cnt_end = it_cnt_begin + n_coord;
    v_coord_cell_offset.resize(n_coord);
    v_coord_cell_offset[0] = 0;
    auto const it_index = v_point_cell_index.begin();
    auto it = thrust::copy_if(it_cnt_begin + 1, it_cnt_end, v_coord_cell_offset.begin() + 1, [=]__device__(int const &i) -> bool {
       return *(it_index + i - 1) != *(it_index + i);
    });
    v_coord_cell_offset.resize(thrust::distance(v_coord_cell_offset.begin(), it));
    v_coord_cell_index.resize(v_coord_cell_offset.size());
    v_coord_cell_size.resize(v_coord_cell_offset.size());
    auto const it_offset = v_coord_cell_offset.begin();
    it_cnt_end = it_cnt_begin + v_coord_cell_offset.size();
    thrust::transform(it_cnt_begin, it_cnt_end - 1, v_coord_cell_size.begin(), [=]__device__(int const &i) -> int {
        return *(it_offset+i+1) - *(it_offset+i);
    });
    v_coord_cell_size[v_coord_cell_size.size()-1] = n_coord - v_coord_cell_offset[v_coord_cell_size.size()-1];
    thrust::transform(it_cnt_begin, it_cnt_end, v_coord_cell_index.begin(), [=]__device__(int const &i) -> int {
        return *(it_index + *(it_offset + i));
    });
//    int sum = 0;
//    for (int i = 0; i < v_coord_cell_size.size(); ++i) {
//        sum += v_coord_cell_size[i];
//    }
//    std::cout << "sum: " << sum << std::endl;
//
//
    std::cout << "v_coord_cell_index begin: " <<  v_coord_cell_index[0] << ", " << v_coord_cell_index[1] << ", " << v_coord_cell_index[2] << std::endl;
    std::cout << "v_coord_cell_index end: " <<  v_coord_cell_index[v_coord_cell_index.size()-1] << ", " <<
        v_coord_cell_index[v_coord_cell_index.size()-2] << ", " << v_coord_cell_index[v_coord_cell_index.size()-3] << std::endl;
}

void nc_tree::collect_cells_in_reach(d_vec<int> &v_point_index, d_vec<int> &v_cell_reach,
        d_vec<int> &v_point_reach_offset, d_vec<int> &v_point_reach_size) noexcept {
    int const n_points = v_point_index.size();
    int const dim_part = v_dim_part_size[0];
    d_vec<int> v_point_reach_full(9 * n_points, -1);

    thrust::counting_iterator<int> it_cnt_begin(0);
    thrust::counting_iterator<int> it_cnt_end = it_cnt_begin + n_points;

    d_vec<int> v_lower_bound(n_points * 3);
    auto it_perm_begin = thrust::make_permutation_iterator(v_point_index.begin(), it_cnt_begin);
    auto it_perm_end = thrust::make_permutation_iterator(v_point_index.end(), it_cnt_end);
    auto it_trans_begin_1 = thrust::make_transform_iterator(it_perm_begin, thrust::placeholders::_1 - 1);
    auto it_trans_end_1 = thrust::make_transform_iterator(it_perm_end, thrust::placeholders::_1 - 1);
    auto it_trans_out_1 = thrust::make_transform_iterator(it_cnt_begin, thrust::placeholders::_1 * 3);
    auto it_perm_out_1 = thrust::make_permutation_iterator(v_lower_bound.begin(), it_trans_out_1);
    thrust::lower_bound(v_coord_cell_index.begin(), v_coord_cell_index.end(), it_trans_begin_1, it_trans_end_1, it_perm_out_1);

    auto it_trans_begin_2 = thrust::make_transform_iterator(it_perm_begin, thrust::placeholders::_1 - dim_part - 1);
    auto it_trans_end_2 = thrust::make_transform_iterator(it_perm_end, thrust::placeholders::_1 - dim_part - 1);
    auto it_trans_out_2 = thrust::make_transform_iterator(it_cnt_begin, (thrust::placeholders::_1 * 3) + 1);
    auto it_perm_out_2 = thrust::make_permutation_iterator(v_lower_bound.begin(), it_trans_out_2);
    thrust::lower_bound(v_coord_cell_index.begin(), v_coord_cell_index.end(), it_trans_begin_2, it_trans_end_2, it_perm_out_2);

    auto it_trans_begin_3 = thrust::make_transform_iterator(it_perm_begin, thrust::placeholders::_1 + dim_part - 1);
    auto it_trans_end_3 = thrust::make_transform_iterator(it_perm_end, thrust::placeholders::_1 + dim_part - 1);
    auto it_trans_out_3 = thrust::make_transform_iterator(it_cnt_begin, (thrust::placeholders::_1 * 3) + 2);
    auto it_perm_out_3 = thrust::make_permutation_iterator(v_lower_bound.begin(), it_trans_out_3);
    thrust::lower_bound(v_coord_cell_index.begin(), v_coord_cell_index.end(), it_trans_begin_3, it_trans_end_3, it_perm_out_3);

    auto const it_coord_index = v_coord_cell_index.begin();
    auto const it_value = it_perm_begin;
    auto const it_bounds = v_lower_bound.begin();
    auto const it_full_reach = v_point_reach_full.begin();
    auto const it_reach_size = v_point_reach_size.begin();
    thrust::for_each(it_cnt_begin, it_cnt_end, [=]__device__(int const &i) -> void {
        auto it_begin = it_full_reach + (i * 9);
        auto it_out = it_begin;
        int val = *(it_bounds + (i * 3));
        if (*(it_coord_index + val) == *(it_value + i) - 1) {
            *(it_out++) = val;
            ++val;
        }
        if (*(it_coord_index + val) == *(it_value + i)) {
            *(it_out++) = val;
            ++val;
        }
        if (*(it_coord_index + val) == *(it_value + i) + 1) {
            *(it_out++) = val;
            ++val;
        }
        val = *(it_bounds + (i * 3) + 1);
        if (*(it_coord_index + val) == *(it_value + i) - dim_part - 1) {
            *(it_out++) = val;
            ++val;
        }
        if (*(it_coord_index + val) == *(it_value + i) - dim_part) {
            *(it_out++) = val;
            ++val;
        }
        if (*(it_coord_index + val) == *(it_value + i) - dim_part + 1) {
            *(it_out++) = val;
            ++val;
        }
        val = *(it_bounds + (i * 3) + 2);
        if (*(it_coord_index + val) == *(it_value + i) + dim_part - 1) {
            *(it_out++) = val;
            ++val;
        }
        if (*(it_coord_index + val) == *(it_value + i) + dim_part) {
            *(it_out++) = val;
            ++val;
        }
        if (*(it_coord_index + val) == *(it_value + i) + dim_part + 1) {
            *(it_out++) = val;
            ++val;
        }
        *(it_reach_size + i) = it_out - it_begin;
    });
    v_cell_reach.resize(v_point_reach_full.size());
    thrust::exclusive_scan(v_point_reach_size.begin(), v_point_reach_size.end(), v_point_reach_offset.begin());
    auto it = thrust::copy_if(v_point_reach_full.begin(), v_point_reach_full.end(), v_cell_reach.begin(), []__device__(int const &val) -> bool {
        return val >= 0;
    });
    v_cell_reach.resize(thrust::distance(v_cell_reach.begin(), it));
#ifdef DEBUG_ON
    std::cout << "v_cell_reach: " << v_cell_reach.size() << std::endl;
#endif
}

void nc_tree::process_points(d_vec<int> &v_point_id, d_vec<float> &v_point_data) noexcept {
    auto const it_status = v_coord_status.begin();
    thrust::for_each(v_point_id.begin(), v_point_id.end(), [=]__device__(int const &id) -> void {
        if (id >= 0) {
            // PROCESSED
            *(it_status + id) = 1;
        }
    });
    d_vec<int> v_point_index(v_point_id.size());
    index_points(v_point_data, v_point_index);
    // obtain reach
    d_vec<int> v_point_cells_in_reach;
    d_vec<int> v_point_cell_reach_offset(v_point_id.size());
    d_vec<int> v_point_cell_reach_size(v_point_id.size());

    collect_cells_in_reach(v_point_index, v_point_cells_in_reach, v_point_cell_reach_offset, v_point_cell_reach_size);

    d_vec<int> v_points_in_reach_size(v_point_id.size());
    d_vec<int> v_points_in_reach_offset(v_point_id.size());
    thrust::counting_iterator<int> it_point_begin(0);
    thrust::counting_iterator<int> it_point_end = it_point_begin + v_point_id.size();

    auto const it_coord_cell_size = v_coord_cell_size.begin();
    auto const it_point_cell_reach = v_point_cell_reach_size.begin();
    auto const it_point_cells_in_reach = v_point_cells_in_reach.begin();
    auto const it_point_cell_reach_offset = v_point_cell_reach_offset.begin();

    thrust::transform(it_point_begin, it_point_end, v_points_in_reach_size.begin(), [=]__device__(int const &i) -> int {
        auto p_sum = 0;
        for (int j = 0; j < *(it_point_cell_reach + i); ++j) {
            p_sum += *(it_coord_cell_size + *(it_point_cells_in_reach + *(it_point_cell_reach_offset + i) + j));
        }
        return p_sum;
    });
    thrust::exclusive_scan(v_points_in_reach_size.begin(), v_points_in_reach_size.end(), v_points_in_reach_offset.begin());
    auto table_size = thrust::reduce(v_points_in_reach_size.begin(), v_points_in_reach_size.end(), 0);
    std::cout << "table_size: " << table_size << std::endl;
    d_vec<int> v_hit_table_id_1(table_size, -1);
    d_vec<int> v_hit_table_id_2(table_size, -1);
    auto const it_hit_table_1 = v_hit_table_id_1.begin();
    auto const it_hit_table_2 = v_hit_table_id_2.begin();
    auto const it_points_in_reach_offset = v_points_in_reach_offset.begin();
    auto const it_points_in_reach_size = v_points_in_reach_size.begin();
    auto const it_coord_id = v_coord_id.begin();
    auto const it_coord_offset = v_coord_cell_offset.begin();
    thrust::for_each(it_point_begin, it_point_end, [=]__device__(int const &i) -> void {
        for (int j = 0; j < *(it_points_in_reach_size + i); ++j) {
            *(it_hit_table_1 + *(it_points_in_reach_offset + i) + j) = i;
            *(it_hit_table_2 + *(it_points_in_reach_offset + i) + j) = *(it_coord_id + *(it_coord_offset + *(it_point_cells_in_reach + i)) + j);
//            v_hit_table_id_2[v_cell_reach_offset[i] + j] = v_coord_id[v_coord_cell_offset[v_point_cells_in_reach[i]] + j];
        }
    });
    auto cnt = thrust::count_if(v_hit_table_id_1.begin(), v_hit_table_id_1.end(), []__device__(int const &v) -> bool {
       return v == -1;
    });
    std::cout << "count if: " << cnt << std::endl;
    auto pair = thrust::minmax_element(v_hit_table_id_1.begin(), v_hit_table_id_1.end());
    std::cout << "min: " << *pair.first << " max: " << *pair.second << std::endl;
    std::cout << "point data size: " << v_point_data.size() << std::endl;

    /*
    s_vec<int> v_cell_reach_size(v_point_cells_in_reach.size());
    s_vec<int> v_cell_reach_offset(v_point_cells_in_reach.size());
    s_vec<int> v_cell_reach_iota(v_point_cells_in_reach.size());
    exa::iota(v_cell_reach_iota, 0, v_cell_reach_iota.size(), 0);
    exa::transform(v_point_cells_in_reach, v_cell_reach_size, 0, v_point_cells_in_reach.size(), 0,
            [&](int const &c_id) -> int {
                return v_coord_cell_size[c_id];
            });
    exa::exclusive_scan(v_cell_reach_size, v_cell_reach_offset, 0, v_cell_reach_size.size(), 0, 0);
    exa::for_each(v_cell_reach_iota, 0, v_cell_reach_iota.size(), [&](int const &i) -> void {
        for (int j = 0; j < v_cell_reach_size[i]; ++j) {
            v_hit_table_id_2[v_cell_reach_offset[i] + j] = v_coord_id[v_coord_cell_offset[v_point_cells_in_reach[i]] + j];
        }
    });
     */


    auto hit1 = thrust::reduce(v_hit_table_id_1.begin(), v_hit_table_id_1.end(), 0);
    auto hit2 = thrust::reduce(v_hit_table_id_2.begin(), v_hit_table_id_2.end(), 0);
    std::cout << "hit1: " << hit1 << " hit2: " << hit2 << std::endl;

    auto const it_coord_data = v_device_coord.begin();
    auto const it_point_data = v_point_data.begin();
    thrust::counting_iterator<int> it_table_cnt_begin(0);
    thrust::counting_iterator<int> it_table_cnt_end = it_table_cnt_begin + v_hit_table_id_1.size();
    auto it_perm_begin_1 = thrust::make_permutation_iterator(v_point_data.begin(), it_table_cnt_begin);
    auto it_trans_begin_1 = thrust::make_transform_iterator(it_perm_begin_1, (thrust::placeholders::_1 * n_dim));

//    auto it_trans_begin = thrust::make_transform_iterator(it_table_cnt_begin, (thrust::placeholders::_1 * n_dim));
//    auto it_trans_end = thrust::make_transform_iterator(it_table_cnt_end, (thrust::placeholders::_1 * n_dim));
//    auto it_trans_begin_1 = thrust::make_transform_iterator(v_hit_table_id_1.begin(), (thrust::placeholders::_1 * n_dim));
//    auto it_trans_end_1 = thrust::make_transform_iterator(v_hit_table_id_1.end(), (thrust::placeholders::_1 * n_dim));
//    auto it_perm_begin_1 = thrust::make_permutation_iterator(v_point_data.begin(), it_trans_begin_1);
//    auto it_perm_end_1 = thrust::make_permutation_iterator(v_point_data.end(), it_trans_end_1);
//    auto it_trans_begin_2 = thrust::make_transform_iterator(v_hit_table_id_2.begin(), (thrust::placeholders::_1 * n_dim));
//    auto it_trans_end_2 = thrust::make_transform_iterator(v_hit_table_id_2.end(), (thrust::placeholders::_1 * n_dim));
//    auto it_perm_begin_2 = thrust::make_permutation_iterator(v_device_coord.begin(), it_trans_begin_2);
//    auto it_perm_end_2 = thrust::make_permutation_iterator(v_device_coord.end(), it_trans_end_2);
//    thrust::make_zip_iterator(thrust::make_tuple(int_in.begin(), float_in.begin())),
    float const _e2 = e2;
    float const _n_dim = n_dim;
    d_vec<float> v_result_table(table_size, 0);
    std::cout << "CHECKPOINT" << std::endl;
    thrust::for_each(it_table_cnt_begin, it_table_cnt_end, [=]__device__(int const &i) -> void {
        auto it_point = it_point_data + (*(it_hit_table_1 + i) * _n_dim);
//        auto it_coord = it_coord_data + (*(it_hit_table_2 + i) * _n_dim);
        float result = 0;
//        #pragma unroll
        for (int d = 0; d < _n_dim; ++d) {
//            result += (*(it_point + d) - *(it_coord + d)) * (*(it_point + d) - *(it_coord + d));
//            result += *(it_point + d);
        }
        if (result > _e2) {
            *(it_hit_table_1 + i) = -1;
        }
    });


//    d_vec<float> v_hit_table_result(table_size);
//    thrust::transform(it_table_cnt_begin, it_table_cnt_end, v_hit_table_result.begin(), [=]__device__(int const &i) -> float {
//        d_vec<float>::iterator it_coord_1 = it_coord_data + (*(it_hit_table_1 + i) * n_dim);
//        d_vec<float>::iterator it_coord_2 = it_coord_data + (*(it_hit_table_2 + i) * n_dim);
//        float result = 0;
//        #pragma unroll
//        for (auto d = 0; d < n_dim; ++d) {
//            result += (*(it_coord_1 + d) - *(it_coord_2 + d)) * (*(it_coord_1 + d) - *(it_coord_2 + d));
//            result += (*(it_coord_1));// - *(it_coord_2)) * (*(it_coord_1) - *(it_coord_2));
//            result += 0.5 * 0.3;
//        }
//        return result;
//        return result <= e2? 1 : -1;
//        *(it_hit_table_1 + i) = -1;
//        if (result > e2) {
//            *(it_hit_table_1 + i) = -1;
//        }
//        return result;
//    });
    std::cout << "CHECKPOINT 2" << std::endl;
    d_vec<int> v_point_nn(v_point_id.size(), 0);
    auto it_point_id = v_point_id.begin();
    auto it_coord_nn = v_coord_nn.begin();
    thrust::transform(it_point_begin, it_point_end, v_point_nn.begin(), [=]__device__(int const &i) -> int {
        int p_m = 0;
        for (int j = 0; j < *(it_points_in_reach_size + i); ++j) {
            if (*(it_hit_table_1 + *(it_points_in_reach_offset + i) + j) != -1) {
                ++p_m;
            }
        }
        if (*(it_point_id + i) >= 0) {
            *(it_coord_nn + *(it_point_id + i)) = p_m;
        }
        return p_m;
    });
    pair = thrust::minmax_element(v_coord_nn.begin(), v_coord_nn.end());
    std::cout << "min_nn: " << *pair.first << " max_nn: " << *pair.second << std::endl;
    /*
    s_vec<int> v_hit_table_iota(v_hit_table_id_1.size());
    exa::iota(v_hit_table_iota, 0, v_hit_table_iota.size(), 0);
    s_vec<int> v_point_nn(v_point_iota.size(), 0);
    exa::for_each(v_hit_table_iota, 0, v_hit_table_iota.size(), [&](int const &i) -> void {
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

    print_cuda_memory_usage();
    /*
    // TODO combine in int64
    exa::for_each(v_point_iota, 0, v_point_iota.size(), [&](int const &i) -> void {
        std::fill(std::next(v_hit_table_id_1.begin(), v_points_in_reach_offset[i]),
                std::next(v_hit_table_id_1.begin(), v_points_in_reach_offset[i] + v_points_in_reach_size[i]),
                i);
    });
    // Make cell offset and size
    s_vec<int> v_cell_reach_size(v_point_cells_in_reach.size());
    s_vec<int> v_cell_reach_offset(v_point_cells_in_reach.size());
    s_vec<int> v_cell_reach_iota(v_point_cells_in_reach.size());
    exa::iota(v_cell_reach_iota, 0, v_cell_reach_iota.size(), 0);
    exa::transform(v_point_cells_in_reach, v_cell_reach_size, 0, v_point_cells_in_reach.size(), 0,
            [&](int const &c_id) -> int {
                return v_coord_cell_size[c_id];
            });
    exa::exclusive_scan(v_cell_reach_size, v_cell_reach_offset, 0, v_cell_reach_size.size(), 0, 0);
    exa::for_each(v_cell_reach_iota, 0, v_cell_reach_iota.size(), [&](int const &i) -> void {
        for (int j = 0; j < v_cell_reach_size[i]; ++j) {
            v_hit_table_id_2[v_cell_reach_offset[i] + j] = v_coord_id[v_coord_cell_offset[v_point_cells_in_reach[i]] + j];
        }
    });

    s_vec<int> v_hit_table_iota(v_hit_table_id_1.size());
    exa::iota(v_hit_table_iota, 0, v_hit_table_iota.size(), 0);
    s_vec<int> v_point_nn(v_point_iota.size(), 0);
    exa::for_each(v_hit_table_iota, 0, v_hit_table_iota.size(), [&](int const &i) -> void {
        if (!dist_leq(&v_point_data[v_hit_table_id_1[i]*n_dim], &v_coord[v_hit_table_id_2[i]*n_dim], n_dim, e2)) {
            v_hit_table_id_2[i] = -1;
        } else {
            ++v_point_nn[v_hit_table_id_1[i]];
//            if (v_coord_status[v_hit_table_id_2[i]] != PROCESSED) {
//                ++v_coord_nn[v_hit_table_id_2[i]];
//            }
        }
    });

    s_vec<int> v_point_cluster(v_point_id.size(), NO_CLUSTER);
    s_vec<int> v_point_new_cluster_mark(v_point_id.size(), 0);
    s_vec<int> v_point_new_cluster_offset(v_point_id.size());

    exa::for_each(v_point_iota, 0, v_point_iota.size(), [&](int const &i) -> void {
        if (v_point_nn[i] >= m) {
            v_point_cluster[i] = i + cluster_size;
            if (v_point_id[i] >= 0) {
                v_coord_nn[v_point_id[i]] = v_point_nn[i];
                if (v_coord_cluster[v_point_id[i]] == NO_CLUSTER) {
                    v_coord_cluster[v_point_id[i]] = v_point_cluster[i];
                } else {
                    v_point_cluster[i] = v_coord_cluster[v_point_id[i]];
//                    std::cout << "CHECKPOINT" << std::endl;
                }
            }
        }
    });

    bool is_done = false;
    int iter_cnt = 0;
    while (!is_done) {
        is_done = true;
        ++iter_cnt;
        exa::for_each(v_point_iota, 0, v_point_iota.size(), [&](int const &i) -> void {
            if (v_point_nn[i] >= m) {
                for (int j = 0; j < v_points_in_reach_size[i]; ++j) {
                    auto id2 = v_hit_table_id_2[v_points_in_reach_offset[i] + j];
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
                            is_done = false;
                        }
                    }
                }
            }
        });
    }
    std::cout << "label iterations: " << iter_cnt << std::endl;
    int new_clusters = 0;
    exa::for_each(v_point_iota, 0, v_point_iota.size(), [&](int const &i) -> void {
        if (v_point_nn[i] >= m && v_point_cluster[i] == i + cluster_size) {
            ++new_clusters;
        }
    });
    std::cout << "new clusters: " << new_clusters << std::endl;
    cluster_size += new_clusters;


    exa::for_each(v_point_iota, 0, v_point_iota.size(), [&](int const &i) -> void {
        if (v_point_nn[i] >= m) {
//            assert(v_point_cluster[i] != NO_CLUSTER);
            for (int j = 0; j < v_points_in_reach_size[i]; ++j) {
                auto id2 = v_hit_table_id_2[v_points_in_reach_offset[i] + j];
                if (id2 == -1) continue;
                if (v_coord_cluster[id2] == NO_CLUSTER) {
                    v_coord_cluster[id2] = v_point_cluster[i];
                }
                else if (v_coord_cluster[id2] != v_point_cluster[i] && v_coord_nn[id2] >= m) {
                    std::cout << "CHECKPINT!!" << std::endl;
//                    assert(v_point_cluster[i] < v_coord_cluster[id2]);
//                    v_coord_cluster[id2] = v_point_cluster[i];
                }
            }
        }
    });
    */
}

void nc_tree::select_and_process() noexcept {
    v_coord_nn.resize(n_coord, 0);
    v_coord_cluster.resize(n_coord, NO_CLUSTER);
    v_coord_status.resize(n_coord, NOT_PROCESSED);

    d_vec<int> v_point_id(v_coord_id.size());
    thrust::sequence(v_point_id.begin(), v_point_id.end(), 0);

    d_vec<int> v_id_chunk;
    d_vec<float> v_data_chunk;
    magma_util::measure_duration("Process Points: ", true, [&]() -> void {
        int n_blocks = 2;
        for (int i = 0; i < 1/*n_blocks*/; ++i) {
            int block_size = magma_util::get_block_size(i, static_cast<int>(v_point_id.size()), n_blocks);
            int block_offset = magma_util::get_block_offset(i, static_cast<int>(v_point_id.size()), n_blocks);
            std::cout << "block offset: " << block_offset << " size: " << block_size << std::endl;
            v_id_chunk.clear();
            v_id_chunk.insert(v_id_chunk.begin(), v_point_id.begin() + block_offset, v_point_id.begin() + block_offset + block_size);
            v_data_chunk.clear();
            v_data_chunk.insert(v_data_chunk.begin(), v_coord.begin() + (block_offset * n_dim), v_coord.begin()
                + ((block_offset + block_size) * n_dim));
            process_points(v_id_chunk, v_data_chunk);

        }
    });

}

void nc_tree::get_result_meta(int &cores, int &noise, int &clusters, int &n) noexcept {
    n = n_coord;

    auto const _m = m;
    cores = thrust::count_if(v_coord_nn.begin(), v_coord_nn.end(), [=]__device__(int const &v) -> bool {
        return v >= _m;
    });
    /*
    int cluster_points = 0;
    for (auto const &cluster : v_coord_cluster) {
        if (cluster >= 0) ++cluster_points;
    }
    noise = n_coord - cluster_points;

    std::unordered_map<int, int> v_cluster_map;
    for (int const &cluster : v_coord_cluster) {
        if (cluster >= 0) {
            auto elem = v_cluster_map.find(cluster);
            if (elem == v_cluster_map.end()) {
                v_cluster_map.insert(std::make_pair(cluster, 1));
            } else {
                (*elem).second++;
            }
        }
    }
    clusters = v_cluster_map.size();
     */
}

void nc_tree::process6() noexcept {
    /*
    v_dim_part_size.resize(2);
    v_dim_part_size[0] = (v_max_bounds[v_dim_order[0]] - v_min_bounds[v_dim_order[0]]) / e + 1;
    v_dim_part_size[1] = (v_max_bounds[v_dim_order[1]] - v_min_bounds[v_dim_order[1]]) / e + 1;
    magma_util::print_v("v_dim_part_size: ", &v_dim_part_size[0], v_dim_part_size.size());
    if (static_cast<uint64_t>(v_dim_part_size[0]) * v_dim_part_size[1] > INT32_MAX) {
        std::cerr << "FAIL: The epsilon value is too low and therefore not supported by the current version for the"
                     " input dataset" << std::endl;
        exit(EXIT_FAILURE);
    }
    magma_util::measure_duration("Cell Indexing: ", true, [&]() -> void {
        index_into_cells(v_coord_id, v_coord_cell_size, v_coord_cell_offset, v_coord_cell_index, v_dim_part_size[0]);
    });
    s_vec<int> v_point_id(v_coord_id.size());
    exa::iota(v_point_id, 0, v_point_id.size(), 0);
    s_vec<int> v_id_chunk;
    s_vec<float> v_data_chunk;
    magma_util::measure_duration("Process Points: ", true, [&]() -> void {
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
            process_points(v_id_chunk, v_data_chunk);

        }
    });

    int cores = 0;
    for (auto const &nn : v_coord_nn) {
        if (nn >= m) ++cores;
    }
    std::cout << "cores: " << cores << std::endl;

    int cluster_points = 0;
    for (auto const &cluster : v_coord_cluster) {
        if (cluster >= 0) ++cluster_points;
    }
    std::cout << "points in cluster: " << cluster_points << std::endl;
    std::cout << "points NOT in cluster: " << v_coord_cluster.size()-cluster_points << std::endl;

    std::unordered_map<int, int> v_cluster_map;
    for (int const &cluster : v_coord_cluster) {
        if (cluster >= 0) {
            auto elem = v_cluster_map.find(cluster);
            if (elem == v_cluster_map.end()) {
                v_cluster_map.insert(std::make_pair(cluster, 1));
            } else {
                (*elem).second++;
            }
        }
    }
    std::cout << "Total number of clusters: " << v_cluster_map.size();
     */
}
