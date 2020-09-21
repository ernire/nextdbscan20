//
// Created by Ernir Erlingsson on 19.8.2020.
//

#ifndef NEXTDBSCAN20_NC_TREE_H
#define NEXTDBSCAN20_NC_TREE_H

#include "magma_util.h"

static const int NO_CLUSTER = -2;
static const int NOT_PROCESSED = -2;
static const int SKIPPED = -1;
static const int MARKED = 0;
static const int PROCESSED = 1;


class nc_tree {
private:
    int const m, n_dim;
    unsigned long const n_coord;
    float const e, e2, e_l;
//    s_vec<int> v_coord_index;
    s_vec<int> v_dim_order;

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
    s_vec<float> v_coord;
    s_vec<float> v_min_bounds, v_max_bounds;
    // TODO optimize sizes
    s_vec<int> v_coord_id;
    s_vec<int> v_coord_cell_index;
    s_vec<int> v_coord_cell_offset;
    s_vec<int> v_coord_cell_size;
    s_vec<int> v_coord_nn;
    s_vec<int> v_coord_status;
    s_vec<int> v_coord_cluster;
    s_vec<int> v_dim_part_size;
    int cluster_size = 0;

    explicit nc_tree(s_vec<float> &v_coord, int const m, float const e, int const n_dim) : v_coord(std::move(v_coord)),
        n_coord(v_coord.size()/n_dim), n_dim(n_dim), m(m), e(e), e2(e*e), e_l(get_lowest_e(e, n_dim)) {
        v_coord_nn.resize(n_coord, 0);
        v_coord_cluster.resize(n_coord, NO_CLUSTER);
        v_coord_status.resize(n_coord, NOT_PROCESSED);
    }

    void process6() noexcept;

    void determine_data_bounds() noexcept;

    void index_into_cells(s_vec<int> &v_point_id, s_vec<int> &v_cell_size, s_vec<int> &v_cell_offset,
            s_vec<int> &v_cell_index, int const dim_part_size) noexcept;

    void process_points(s_vec<int> &v_point_id, s_vec<float> &v_point_data) noexcept;

};


#endif //NEXTDBSCAN20_NC_TREE_H
