//
// Created by Ernir Erlingsson on 19.8.2020.
//

#ifndef NEXTDBSCAN20_NC_TREE_H
#define NEXTDBSCAN20_NC_TREE_H

static const int NO_CLUSTER = -2;
static const int NOT_PROCESSED = -2;
static const int SKIPPED = -1;
static const int MARKED = 0;
static const int PROCESSED = 1;


#ifdef CUDA_ON
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
template <typename T>
using s_vec = thrust::host_vector<T>;
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

class nc_tree {
private:
    int const m, n_dim;
    unsigned long const n_coord;
    float const e, e2, e_l;
//    s_vec<int> v_coord_index;


    static float get_lowest_e(float const e, long const n_dim) noexcept {
        // TODO find a less wasteful formula to maintain precision
        return e / 2;
//        return e / sqrtf(3);
        /*
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
         */

    }

    inline static bool dist_leq(float const *coord1, float const *coord2, int const n_dim, float const e2) noexcept {
        float tmp = 0;
//    #pragma omp simd
        for (auto d = 0; d < n_dim; d++) {
            tmp += (coord1[d] - coord2[d]) * (coord1[d] - coord2[d]);
        }
        return tmp <= e2;
    }

    inline static int cell_index(float const coord, float const bound, float const e) noexcept {
        return (int)(((coord - bound) / e));
    }

    void collect_cells_in_reach(s_vec<int> &v_point_index,s_vec<int> &v_cell_reach,
            s_vec<int> &v_point_reach_offset, s_vec<int> &v_point_reach_size) noexcept;

public:
    h_vec<float> v_coord;
#ifdef CUDA_ON
    d_vec<float> v_device_coord;
#endif
    d_vec<float> v_min_bounds, v_max_bounds;
    d_vec<int> v_dim_order;
    d_vec<int> v_coord_id;
    // TODO optimize sizes
    d_vec<int> v_coord_cell_index;
    d_vec<int> v_coord_cell_offset;
    d_vec<int> v_coord_cell_size;
    d_vec<int> v_coord_nn;
    d_vec<int> v_coord_status;
    d_vec<int> v_coord_cluster;
    d_vec<int> v_dim_part_size;
    int cluster_size = 0;

    explicit nc_tree(h_vec<float> &v_coord, int const m, float const e, int const n_dim) : v_coord(std::move(v_coord)),
        m(m), n_dim(n_dim), n_coord(v_coord.size()/n_dim), e(e), e2(e*e), e_l(get_lowest_e(e, n_dim)) {
#ifdef CUDA_ON
        v_device_coord = v_coord;
#endif
    }

    void process6() noexcept;

    void determine_data_bounds() noexcept;

    void initialize_cells() noexcept;

    void index_into_cells(s_vec<int> &v_point_id, s_vec<int> &v_cell_size, s_vec<int> &v_cell_offset,
            s_vec<int> &v_cell_index, int const dim_part_size) noexcept;
//
    void process_points(d_vec<int> &v_point_id, d_vec<float> &v_point_data) noexcept;

    void select_and_process() noexcept;

    void get_result_meta(int &cores, int &noise, int &clusters, int &n) noexcept;

};


#endif //NEXTDBSCAN20_NC_TREE_H
