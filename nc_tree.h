//
// Created by Ernir Erlingsson on 19.8.2020.
//

#ifndef NEXTDBSCAN20_NC_TREE_H
#define NEXTDBSCAN20_NC_TREE_H

#include "magma_meta.h"
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
//    s_vec<int> v_sorted_coord_id;
    s_vec<int> v_coord_index;
    s_vec<int> v_dim_order;

    static void index_coords(s_vec<float> &v_coord, s_vec<float> &v_min_bound, s_vec<int> &v_coord_index,
            float const e, int const n_dim) noexcept {
#ifdef DEBUG_ON
        assert(v_coord.size() == v_coord_index.size());
#endif
        s_vec<int> v_id(v_coord.size());
        exa::iota(v_id, 0, v_id.size(), 0);
        exa::transform(v_id, v_coord_index, 0, v_id.size(), 0, [&](int const &i) -> int {
//            return get_cell_index(v_coord[i], v_min_bound[i%n_dim], e);
        });
    }

    static void sort_and_count(s_vec<float> &v_coord, s_vec<int> &v_coord_id, s_vec<int> &v_coord_index,
            s_vec<float> &v_min_bounds, s_vec<int> &v_offset, s_vec<int> &v_size, int n_dim, float e) noexcept;

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
//    inline static bool are_in_range(s_vec<int>::iterator &it1, s_vec<int>::iterator &it2, int const n_dim) {
//        for (int d = 0; d < n_dim; ++d) {
//            if (abs(*(it1+d) - *(it2+d)) > 2)
//                return false;
//        }
//        return true;
//    }

    static s_vec<int>::iterator lower_cell_bound(s_vec<int>::iterator &begin, s_vec<int>::iterator &end,
            int const val) noexcept {
        /*
        if (end - begin < 10) {
            auto end2 = end;
            while (end2 != begin) {
                if (*(end2-1) < val)
                    return end2;
                --end2;
            }
            return begin;
        }
         */
        return std::lower_bound(begin, end, val, [&](auto const &o, int const &val) -> bool {
            return o < val;
        });
    }

    static s_vec<int>::iterator upper_cell_bound(s_vec<int>::iterator &begin, s_vec<int>::iterator &end,
            int const val) noexcept {
        /*
        if (end - begin < 10) {
            auto begin2 = begin;
            while (begin2 != end) {
                if (*begin2 > val)
                    return begin2;
                ++begin2;
            }
            return end;
        }
         */
        return std::upper_bound(begin, end, val, [&](int const &val, auto const &o) -> bool {
            return o > val;
        });
    }

    inline static int cell_index(float const coord, float const bound, float const e) noexcept {
        return (int)(((coord - bound) / e));
    }

    inline static s_vec<int>::iterator get_dim_iter(s_vec<int> &v_coord_index, s_vec<int> &v_sorted_coord_id,
            int const offset, int const n_dim) {
        return std::next(v_coord_index.begin(), v_sorted_coord_id[offset]*n_dim);
    }

//    inline int get_dim_index(int const offset, int const d) {
//        assert(offset >= 0 && offset < v_sorted_coord_id.size());
//        return v_coord_index[(v_sorted_coord_id[offset]*n_dim)+d];

//    }

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

    void determine_cell_reach(s_vec<int> &v_cell_index, s_vec<int> &v_cell_reach, s_vec<int> &v_cell_reach_size,
            s_vec<int> &v_cell_offset, int const n_cells, s_vec<int> &v_dim_part_size) noexcept;

    void determine_data_bounds() noexcept;

    void index_into_cells(s_vec<int> &v_point_id, s_vec<int> &v_cell_size, s_vec<int> &v_cell_offset,
            s_vec<int> &v_cell_index, int const dim_part_size) noexcept;

    void index_points(s_vec<int> &v_p_id, s_vec<int> &v_p_cell_index) noexcept;

    void process_points(s_vec<int> &v_point_id, s_vec<float> &v_point_data) noexcept;

};


#endif //NEXTDBSCAN20_NC_TREE_H
