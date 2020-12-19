//
// Created by Ernir Erlingsson on 19.8.2020.
//

#ifndef NEXTDBSCAN20_DATA_PROCESS_H
#define NEXTDBSCAN20_DATA_PROCESS_H

#ifdef CUDA_ON
__device__
#endif
static const int NO_CLUSTER = INT32_MAX;
#ifdef CUDA_ON
__device__
#endif
static const int NOT_PROCESSED = -2;
static const int SKIPPED = -1;
#ifdef CUDA_ON
__device__
#endif
static const int MARKED = 0;
#ifdef CUDA_ON
__device__
#endif
static const int PROCESSED = 1;
static const float FLOAT_MAX = 3.40282347e+38F;

#ifdef CUDA_ON
__device__
#endif
static const float _FLOAT_MAX = 3.40282347e+38F;



#ifdef CUDA_ON
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
template <typename T>
using h_vec = thrust::host_vector<T>;
template <typename T>
using d_vec = thrust::device_vector<T>;
#else
#include <vector>
template <typename T>
using s_vec = std::vector<T>;
template <typename T>
using h_vec = std::vector<T>;
template <typename T>
using d_vec = std::vector<T>;
#endif
#include <cmath>
#include "magma_mpi.h"

class data_process {
private:
    int const m, n_dim;
    std::size_t const n_coord;
    float const e, e2;

    static float get_lowest_e(float const e, long const n_dim) noexcept {
        // TODO find a less wasteful formula to maintain precision
//        return e / 2;
//        return e / sqrtf(3);

        if (n_dim <= 3) {
            return e / sqrtf(3);
        } else if (n_dim <= 8) {
            return e / sqrtf(3.5);
        } else if (n_dim <= 30) {
            return e / sqrtf(4);
        } else if (n_dim <= 80) {
            return e / sqrtf(5);
        } else {
            return e / sqrtf(6);
        }


    }

#ifdef CUDA_ON
    __device__
    inline static bool dist_leq(thrust::device_ptr<float> const coord1, thrust::device_ptr<float> const coord2, int const n_dim, float const e2) noexcept {
        float tmp = 0;
        for (auto d = 0; d < n_dim; d++) {
            tmp += (coord1[d] - coord2[d]) * (coord1[d] - coord2[d]);
        }
        return tmp <= e2;
    }
#else
    inline static bool dist_leq(float const *coord1, float const *coord2, int const n_dim, float const e2) noexcept {
        float tmp = 0;
        for (auto d = 0; d < n_dim; d++) {
            tmp += (coord1[d] - coord2[d]) * (coord1[d] - coord2[d]);
        }
        return tmp <= e2;
    }
#endif

public:
    d_vec<float> v_coord;
    d_vec<float> v_min_bounds, v_max_bounds;
    d_vec<int> v_dim_order;
    d_vec<int> v_coord_id;
    // TODO optimize memory usage
    d_vec<int> v_coord_cell_id;
    d_vec<int> v_coord_cell_index;
    d_vec<int> v_coord_cell_offset;
    d_vec<int> v_coord_cell_size;
    d_vec<int> v_coord_nn;
    d_vec<int> v_coord_status;
    d_vec<int> v_coord_cluster_index;
    d_vec<int> v_cluster_label;
    d_vec<int> v_dim_part_size;


    // nc tree
    d_vec<int> v_nc_size;
    d_vec<int> v_nc_offset;
    d_vec<int> v_nc_lvl_size;
    d_vec<int> v_nc_lvl_offset;
#ifdef CUDA_ON
    explicit data_process(h_vec<float> &v_coord, int const m, float const e, int const n_dim)
        : m(m), n_dim(n_dim), n_coord(v_coord.size()/n_dim), e(e), e2(e*e), v_coord(v_coord) {}
#else
    explicit data_process(
            h_vec<float> &v_coord,
            int const m,
            float const e,
            int const n_dim)
        : m(m), n_dim(n_dim), n_coord(v_coord.size()/n_dim), e(e), e2(e*e), v_coord(std::move(v_coord)) {}
#endif

    void collect_cells_in_reach(
            d_vec<long long> const &v_point_index,
            d_vec<int> &v_cell_reach,
            d_vec<int> &v_point_reach_offset,
            d_vec<int> &v_point_reach_size) noexcept;

    void determine_data_bounds() noexcept;

    void initialize_cells() noexcept;

    void build_nc_tree() noexcept;

    void index_points(
            d_vec<float> const &v_data,
            d_vec<long long> &v_index) noexcept;

    void process_points3(
            d_vec<int> const &v_point_id,
            d_vec<float> const &v_point_data,
            d_vec<long long> &v_point_index,
            d_vec<int> &v_point_cells_in_reach,
            d_vec<int> &v_point_cell_reach_offset,
            d_vec<int> &v_point_cell_reach_size,
            magmaMPI mpi) noexcept;

    void process_points2(
            d_vec<int> const &v_point_id,
            d_vec<float> const &v_point_data,
            d_vec<long long> &v_point_index,
            d_vec<int> &v_point_cells_in_reach,
            d_vec<int> &v_point_cell_reach_offset,
            d_vec<int> &v_point_cell_reach_size,
            magmaMPI mpi) noexcept;

    void select_and_process(magmaMPI mpi) noexcept;

    void get_result_meta(
            long long &processed,
            int &cores,
            int &noise,
            int &clusters,
            int &n,
            magmaMPI mpi) noexcept;

};


#endif //NEXTDBSCAN20_DATA_PROCESS_H
