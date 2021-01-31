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
static const int UNASSIGNED = INT32_MAX;
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
static const int PROCESSED = -100;
//static const float FLOAT_MAX = 3.40282347e+38F;

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
    std::size_t const n_total_coord;
    float const e, e2;

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
    d_vec<float> v_cell_AABB;
#ifdef CUDA_ON
    explicit data_process(h_vec<float> &v_coord, int const m, float const e, int const n_dim, int const n_total_coord)
        : m(m), n_dim(n_dim), n_coord(v_coord.size()/n_dim), e(e), e2(e*e), n_total_coord(n_total_coord), v_coord(v_coord) {}
#else
    explicit data_process(
            h_vec<float> &v_coord,
            int const m,
            float const e,
            int const n_dim,
            int const n_total_coord)
        : m(m), n_dim(n_dim), n_coord(v_coord.size()/n_dim), e(e), e2(e*e), n_total_coord(n_total_coord), v_coord(std::move(v_coord)) {}
#endif

    void process_points(d_vec<int> const &v_point_id, d_vec<float> const &v_point_data, d_vec<int> &v_point_nn,
            d_vec<int> &v_tracker, int track_height, magmaMPI mpi) noexcept;

    void determine_data_bounds() noexcept;

    void build_nc_tree() noexcept;

    void index_points(
            d_vec<float> const &v_data,
            d_vec<long long> &v_index) noexcept;

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
