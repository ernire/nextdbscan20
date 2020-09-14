//
// Created by Ernir Erlingsson on 19.8.2020.
//

#include "nc_tree.h"
#include <unordered_set>

//inline int cell_index(s_vec<float> &v_coords, s_vec<float> &v_min_bounds, int const i, int const n_dim,
//        int const d, float const e_l) {
//    return (int)((v_coords[(i*n_dim)+d] - v_min_bounds[d]) / e_l);
//}



void nc_tree::determine_data_bounds() noexcept {
    v_min_bounds.resize(n_dim);
    v_max_bounds.resize(n_dim);
    v_coord_id.resize(n_coord);
    exa::iota(v_coord_id, 0, v_coord_id.size(), 0);
    exa::iota(v_min_bounds, 0, v_min_bounds.size(), 0);
    for (std::size_t d = 0; d < n_dim; ++d) {
        auto minmax = exa::minmax_element(v_coord_id, 0, v_coord_id.size(),
                [&](auto const v1, auto const v2) -> bool {
                    return v_coord[v1 * n_dim + d] < v_coord[v2 * n_dim + d];
                });
        v_min_bounds[d] = v_coord[(minmax.first * n_dim) + d];
        v_max_bounds[d] = v_coord[(minmax.second * n_dim) + d];
    }
    v_dim_order.resize(n_dim);
    exa::iota(v_dim_order, 0, v_dim_order.size(), 0);
    exa::sort(v_dim_order, 0, v_dim_order.size(), [&](int const &d1, int const &d2) -> bool {
       return (v_max_bounds[d1] - v_min_bounds[d1]) > (v_max_bounds[d2] - v_min_bounds[d2]);
    });
    magma_util::print_vector("dim order: ", v_dim_order);
}

void nc_tree::sort_and_count(s_vec<float> &v_coord, s_vec<int> &v_coord_id, s_vec<int> &v_coord_index, s_vec<float> &v_min_bounds,
        s_vec<int> &v_offset, s_vec<int> &v_size, int const n_dim, float const e) noexcept {
    index_coords(v_coord, v_min_bounds, v_coord_index, e, n_dim);

    exa::sort(v_coord_id, 0, v_coord_id.size(), [&](auto const &i1, auto const &i2) -> bool {
        for (int d = 0; d < n_dim; ++d) {
            if (v_coord_index[i1 * n_dim+d] < v_coord_index[i2 * n_dim+d]) {
                return true;
            }
            if (v_coord_index[i1 * n_dim+d] > v_coord_index[i2 * n_dim+d]) {
                return false;
            }
        }
        return false;
    });
    s_vec<int> v_iota(v_coord_id.size());
    exa::iota(v_iota, 0, v_iota.size(), 0);
    exa::unique(v_iota, v_offset, 0, v_iota.size(), 0, [&](auto const &i) -> bool {
        for (auto d = 0; d < n_dim; ++d) {
            auto ci1 = v_coord_index[v_coord_id[i-1]*n_dim+d];
            auto ci2 = v_coord_index[v_coord_id[i]*n_dim+d];
            if (ci1 != ci2)
                return true;
        }
        return false;
    });
    v_size.resize(v_offset.size());
    exa::iota(v_size, 0, v_size.size(), 0);
    exa::transform(v_size, v_size, 0, v_size.size()-1, 0,
            [&](auto const &i) -> int {
                return  v_offset[i+1] - v_offset[i];
            });
    v_size[v_size.size()-1] = (v_coord.size()/n_dim) - v_offset[v_size.size()-1];
}

inline bool is_within_manhattan_2(s_vec<int>::iterator p1, s_vec<int>::iterator p2) {
    if (*p1 - *p2 < -2 || *p1 - *p2 > 2 || *(p1+1) - *(p2+1) < -2 || *(p1+1) - *(p2+1) > 2)
        return false;
//    if ((*p1 -*p2 == -1 || *p1 -*p2 == 1) && ((*(p1+1) - *(p2+1) < -1) || (*(p1+1) - *(p2+1) > 1))) {
//        return false;
//    }
    return true;
}



void nc_tree::index_points(s_vec<int> &v_p_id, s_vec<int> &v_p_cell_index) noexcept {
    exa::for_each(v_p_id, 0, v_p_id.size(), [&](int const &i) -> void {
        v_p_cell_index[i*2] = ((v_coord[i * n_dim + v_dim_order[0]] - v_min_bounds[v_dim_order[0]]) / e_l);
        v_p_cell_index[i*2+1] = ((v_coord[i * n_dim + v_dim_order[1]] - v_min_bounds[v_dim_order[1]]) / e_l);
    });
}

/*
void nc_tree::index_into_cells(s_vec<int> &v_p_id, s_vec<int> &v_c_index, s_vec<int> &v_cell_offset,
        s_vec<int> &v_cell_size) noexcept {
    s_vec<int> v_p_cell_index(v_p_id.size() * 2);
    index_points(v_p_id, v_p_cell_index);
    exa::sort(v_p_id, 0, v_p_id.size(), [&](auto const &i1, auto const &i2) -> bool {
        if (v_p_cell_index[i1*2] < v_p_cell_index[i2*2])
            return true;
        if (v_p_cell_index[i1*2] > v_p_cell_index[i2*2])
            return false;
        if (v_p_cell_index[i1*2+1] < v_p_cell_index[i2*2+1])
            return true;
        return false;
    });

    s_vec<int> v_iota(v_p_id.size());
    exa::iota(v_iota, 0, v_iota.size(), 0);
    exa::unique(v_iota, v_cell_offset, 0, v_iota.size(), 0, [&](auto const &i) -> bool {
        if (v_p_cell_index[v_p_id[i - 1] * 2] != v_p_cell_index[v_p_id[i] * 2]
            || v_p_cell_index[v_p_id[i - 1] * 2 + 1] != v_p_cell_index[v_p_id[i] * 2 + 1])
            return true;
        return false;
    });
    v_cell_size.resize(v_cell_offset.size());
    exa::iota(v_cell_size, 0, v_cell_size.size(), 0);
    exa::transform(v_cell_size, v_cell_size, 0, v_cell_size.size()-1, 0,
            [&](auto const &i) -> int {
                return  v_cell_offset[i+1] - v_cell_offset[i];
            });
    v_cell_size[v_cell_size.size()-1] = n_coord - v_cell_offset[v_cell_size.size()-1];

    v_iota.resize(v_cell_offset.size());
    v_c_index.resize(v_cell_offset.size() * 2);
    exa::for_each(v_iota, 0, v_iota.size(), [&](int const &i) -> void {
        v_c_index[i * 2] = v_p_cell_index[v_p_id[v_cell_offset[i]] * 2];
        v_c_index[i * 2 + 1] = v_p_cell_index[v_p_id[v_cell_offset[i]] * 2 + 1];
    });

#ifdef DEBUG_ON
    std::cout << "Total number of cells: " << v_cell_size.size() << std::endl;
    assert(v_cell_offset.size() == v_cell_size.size());
    for (int c1 = 0; c1 < v_cell_size.size(); c1++) {
        for (int i = 1; i < v_cell_size[c1]; ++i) {
            assert(v_c_index[c1 * 2] == v_p_cell_index[v_p_id[v_cell_offset[c1] + i - 1] * 2]);
            assert(v_c_index[c1 * 2 + 1] == v_p_cell_index[v_p_id[v_cell_offset[c1] + i - 1] * 2 + 1]);
        }
    }
#endif
}
*/

void nc_tree::index_into_cells(s_vec<int> &v_point_id, s_vec<int> &v_cell_size, s_vec<int> &v_cell_offset,
        s_vec<int> &v_cell_index, int const dim_part_size) noexcept {
    auto v_point_cell_index = v_point_id;
    auto v_iota = v_point_id;
    exa::transform(v_point_cell_index, v_point_cell_index, 0, v_point_cell_index.size(), 0,
            [&](int const &id) -> int {
                return cell_index(v_coord[id * n_dim + v_dim_order[0]], v_min_bounds[v_dim_order[0]], e)
                       + (cell_index(v_coord[id * n_dim + v_dim_order[1]], v_min_bounds[v_dim_order[1]], e) * dim_part_size);
            });
    exa::sort(v_point_id, 0, v_point_id.size(), [&](auto const &i1, auto const &i2) -> bool {
        if (v_point_cell_index[i1] < v_point_cell_index[i2])
            return true;
        if (v_point_cell_index[i1] > v_point_cell_index[i2])
            return false;
        return false;
    });
    exa::unique(v_iota, v_cell_offset, 0, v_iota.size(), 0, [&](auto const &i) -> bool {
        if (v_point_cell_index[v_point_id[i]] != v_point_cell_index[v_point_id[i-1]])
            return true;
        return false;
    });
    v_cell_size.resize(v_cell_offset.size());
    exa::iota(v_cell_size, 0, v_cell_size.size(), 0);
    exa::transform(v_cell_size, v_cell_size, 0, v_cell_size.size()-1, 0,
            [&](auto const &i) -> int {
                return  v_cell_offset[i+1] - v_cell_offset[i];
            });
    v_cell_size[v_cell_size.size()-1] = n_coord - v_cell_offset[v_cell_size.size()-1];
    v_iota.resize(v_cell_offset.size());
    v_cell_index = v_iota;
    exa::transform(v_cell_index, v_cell_index, 0, v_cell_index.size(), 0, [&](int const &i) -> int {
        return v_point_cell_index[v_point_id[v_cell_offset[i]]];
    });
}

void nc_tree::determine_cell_reach(s_vec<int> &v_cell_index, s_vec<int> &v_cell_reach, s_vec<int> &v_cell_reach_size,
        s_vec<int> &v_cell_reach_offset, int const n_cells, s_vec<int> &v_dim_part_size) noexcept {
    s_vec<int> v_cell_reach_full(9 * n_cells, -1);
    s_vec<int> v_cell_iota(n_cells);
    v_cell_reach_offset.resize(n_cells);
    v_cell_reach_size.resize(n_cells);
    exa::iota(v_cell_iota, 0, v_cell_iota.size(), 0);
    exa::for_each(v_cell_iota, 0, v_cell_iota.size(), [&](int const &i) -> void {
        auto begin = std::next(v_cell_reach_full.begin(), i * 9);
        auto i_index = begin;
        *(i_index++) = i;
        // middle
        if (v_cell_index[i] % v_dim_part_size[0] > 0 && v_cell_index[i-1] == v_cell_index[i]-1) {
            *(i_index++) = i-1;
//            std::cout << "1: " << v_cell_index[i-1] << " " << v_cell_index[i] << std::endl;
        }
        if (v_cell_index[i] % v_dim_part_size[0] < v_dim_part_size[0]-1 && v_cell_index[i]+1 == v_cell_index[i+1]) {
            *(i_index++) = i+1;
//            std::cout << "2: " << v_cell_index[i] << " " << v_cell_index[i+1] << std::endl;
        }
        // above
        if (v_cell_index[i] >= v_dim_part_size[0]) {
            auto low = std::lower_bound(v_cell_index.begin(), v_cell_index.end(), v_cell_index[i]-v_dim_part_size[0]-1);
            if (*low == v_cell_index[i]-v_dim_part_size[0]-1) {
                *(i_index++) = low - v_cell_index.begin();
                ++low;
            }
            if (*low == v_cell_index[i]-v_dim_part_size[0]) {
                *(i_index++) = low - v_cell_index.begin();
                ++low;
            }
            if (*low == v_cell_index[i]-v_dim_part_size[0]+1) {
                *(i_index++) = low - v_cell_index.begin();
                ++low;
            }
        }
        // below
        if (v_cell_index[i] / v_dim_part_size[0] < v_dim_part_size[1]-1) {
            auto low = std::lower_bound(v_cell_index.begin(), v_cell_index.end(), v_cell_index[i]+v_dim_part_size[0]-1);
            if (*low == v_cell_index[i]+v_dim_part_size[0]-1) {
                *(i_index++) = low - v_cell_index.begin();
                ++low;
            }
            if (*low == v_cell_index[i]+v_dim_part_size[0]) {
                *(i_index++) = low - v_cell_index.begin();
                ++low;
            }
            if (*low == v_cell_index[i]+v_dim_part_size[0]+1) {
                *(i_index++) = low - v_cell_index.begin();
                ++low;
            }
        }
        v_cell_reach_size[i] = i_index - begin;
    });
    exa::exclusive_scan(v_cell_reach_size, v_cell_reach_offset, 0, v_cell_reach_size.size(), 0, 0);
    exa::copy_if(v_cell_reach_full, v_cell_reach, 0, v_cell_reach_full.size(), 0, [&](int const &val) -> bool {
        return val >= 0;
    });
#ifdef DEBUG_ON
    auto sum = std::reduce(v_cell_reach_size.begin(), v_cell_reach_size.end(), 0);
    assert(sum == v_cell_reach.size());
#endif
}

void nc_tree::collect_cells_in_reach(s_vec<int> &v_point_index, s_vec<int> &v_cell_reach,
        s_vec<int> &v_point_reach_offset, s_vec<int> &v_point_reach_size) noexcept {
    int const n_points = v_point_index.size();
    s_vec<int> v_point_reach_full(9 * n_points, -1);
    s_vec<int> v_point_iota(n_points);
    exa::iota(v_point_iota, 0, v_point_iota.size(), 0);
    exa::for_each(v_point_iota, 0, v_point_iota.size(), [&](int const &i) -> void {
        auto begin = std::next(v_point_reach_full.begin(), i * 9);
        auto i_index = begin;
        auto low = std::lower_bound(v_coord_cell_index.begin(), v_coord_cell_index.end(), v_point_index[i]-1);
        if (*low == v_point_index[i]-1) {
            *(i_index++) = low - v_coord_cell_index.begin();
            ++low;
        }
        if (*low == v_point_index[i]) {
            *(i_index++) = low - v_coord_cell_index.begin();
            ++low;
        }
        if (*low == v_point_index[i]+1) {
            *(i_index++) = low - v_coord_cell_index.begin();
            ++low;
        }
        // above
        if (v_point_index[i] >= v_dim_part_size[0]) {
            low = std::lower_bound(v_coord_cell_index.begin(), v_coord_cell_index.end(), v_point_index[i]-v_dim_part_size[0]-1);
            if (*low == v_point_index[i]-v_dim_part_size[0]-1) {
                *(i_index++) = low - v_coord_cell_index.begin();
                ++low;
            }
            if (*low == v_point_index[i]-v_dim_part_size[0]) {
                *(i_index++) = low - v_coord_cell_index.begin();
                ++low;
            }
            if (*low == v_point_index[i]-v_dim_part_size[0]+1) {
                *(i_index++) = low - v_coord_cell_index.begin();
                ++low;
            }
        }
        // below
        if (v_point_index[i] / v_dim_part_size[0] < v_dim_part_size[1]-1) {
            low = std::lower_bound(v_coord_cell_index.begin(), v_coord_cell_index.end(), v_point_index[i]+v_dim_part_size[0]-1);
            if (*low == v_point_index[i]+v_dim_part_size[0]-1) {
                *(i_index++) = low - v_coord_cell_index.begin();
                ++low;
            }
            if (*low == v_point_index[i]+v_dim_part_size[0]) {
                *(i_index++) = low - v_coord_cell_index.begin();
                ++low;
            }
            if (*low == v_point_index[i]+v_dim_part_size[0]+1) {
                *(i_index++) = low - v_coord_cell_index.begin();
                ++low;
            }
        }
        v_point_reach_size[i] = i_index - begin;
    });
    exa::exclusive_scan(v_point_reach_size, v_point_reach_offset, 0, v_point_reach_size.size(), 0, 0);
    exa::copy_if(v_point_reach_full, v_cell_reach, 0, v_point_reach_full.size(), 0, [&](int const &val) -> bool {
        return val >= 0;
    });

}

void nc_tree::process_points(s_vec<int> &v_point_id, s_vec<float> &v_point_data) noexcept {
    std::cout << "id size: " << v_point_id.size() << " : " << v_point_data.size() << std::endl;
    s_vec<int> v_point_iota(v_point_id.size());
    exa::iota(v_point_iota, 0, v_point_iota.size(), 0);
    exa::for_each(v_point_id, 0, v_point_id.size(), [&](int const &id) -> void {
        if (id >= 0) {
            // local, we can modify the status
            v_coord_status[id] = PROCESSED;
        }
    });

    // calculate cell index
    auto v_point_index = v_point_iota;
    exa::transform(v_point_index, v_point_index, 0, v_point_index.size(), 0,
        [&](int const &id) -> int {
            return cell_index(v_point_data[id * n_dim + v_dim_order[0]], v_min_bounds[v_dim_order[0]], e)
                   + (cell_index(v_point_data[id * n_dim + v_dim_order[1]],
                           v_min_bounds[v_dim_order[1]], e) * v_dim_part_size[0]);
        });
    // obtain reach
    s_vec<int> v_point_cells_in_reach(v_point_iota.size());
    s_vec<int> v_point_cell_reach_offset(v_point_iota.size());
    s_vec<int> v_point_cell_reach_size(v_point_iota.size());

    collect_cells_in_reach(v_point_index, v_point_cells_in_reach, v_point_cell_reach_offset, v_point_cell_reach_size);

    s_vec<int> v_points_in_reach_size(v_point_iota.size(), 0);
    s_vec<int> v_points_in_reach_offset(v_point_iota.size());
    // calculate points in reach for each processed point
    exa::for_each(v_point_iota, 0, v_point_iota.size(), [&](int const &i) -> void {
        auto p_sum = 0;
        for (int j = 0; j < v_point_cell_reach_size[i]; ++j) {
            p_sum += v_coord_cell_size[v_point_cells_in_reach[v_point_cell_reach_offset[i] + j]];
        }
        v_points_in_reach_size[i] = p_sum;
    });
    exa::exclusive_scan(v_points_in_reach_size, v_points_in_reach_offset, 0, v_points_in_reach_size.size(), 0, 0);
    long long table_size = exa::reduce(v_points_in_reach_size, 0, v_points_in_reach_size.size(), 0);
    // TODO combine in int64
    s_vec<int> v_hit_table_id_1(table_size, -1);
    s_vec<int> v_hit_table_id_2(table_size, -1);
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

#ifdef DEBUG_ON
    for (auto const &val : v_hit_table_id_1) {
        assert(val != -1);
    }

    for (auto const &val : v_hit_table_id_2) {
        assert(val != -1);
    }
#endif

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
            assert(v_point_cluster[i] != NO_CLUSTER);
            for (int j = 0; j < v_points_in_reach_size[i]; ++j) {
                auto id2 = v_hit_table_id_2[v_points_in_reach_offset[i] + j];
                if (id2 == -1) continue;
                if (v_coord_cluster[id2] == NO_CLUSTER) {
                    v_coord_cluster[id2] = v_point_cluster[i];
                }
//                else if (v_coord_cluster[id2] != v_point_cluster[i] && v_coord_nn[id2] >= m) {
//                    std::cout << "CHECKPINT!!" << std::endl;
//                }
            }
        }
    });

}

void nc_tree::process6() noexcept {
//    s_vec<int> v_dim_part_size(2);
    v_dim_part_size.resize(2);
    v_dim_part_size[0] = (v_max_bounds[v_dim_order[0]] - v_min_bounds[v_dim_order[0]]) / e + 1;
    v_dim_part_size[1] = (v_max_bounds[v_dim_order[1]] - v_min_bounds[v_dim_order[1]]) / e + 1;
    magma_util::print_vector("v_dim_part_size: ", v_dim_part_size);
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
            int block_size = magma_util::get_block_size(i, v_point_id.size(), n_blocks);
            int block_offset = magma_util::get_block_offset(i, v_point_id.size(), n_blocks);
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
}
