//
// Created by Ernir Erlingsson on 14.8.2020.
//

#ifndef EXAFOUNDRY_EXA_H
#define EXAFOUNDRY_EXA_H

#include <vector>
#include <algorithm>
#include <cassert>
#include <vector>
#include <utility>
#include <numeric>
#include <limits>
#include <iostream>
#include <omp.h>
#include "magma_util.h"

template <typename T>
using s_vec = std::vector<T>;

namespace exa {

    template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    void fill(s_vec<T> &v, std::size_t const begin, std::size_t const end, T const val) noexcept {
#ifdef DEBUG_ON
        assert(begin <= end);
#endif
        #pragma omp parallel for
        for (std::size_t i = begin; i < end - begin; ++i) {
            v[i] = val;
        }
#ifdef EXT_DEBUG_ON
        auto v_copy = v;
        std::fill(std::next(v.begin(), begin), std::next(v.begin(), end), val);
        for (int i = begin; i < end - begin; ++i) {
            assert(v[i] == v_copy[i]);
        }
#endif
    }

    template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    void iota(s_vec<T> &v, std::size_t const begin, std::size_t const end, int const startval) noexcept {
#ifdef DEBUG_ON
        assert(begin <= end);
#endif
#pragma omp parallel for
        for (std::size_t i = begin; i < end - begin; ++i) {
            v[i] = startval + i - begin;
        }
    }

    template <typename T, typename F, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    std::size_t count_if(s_vec<T> &v, std::size_t const begin, std::size_t const end, F const &functor) {
#ifdef DEBUG_ON
        assert(begin <= end);
#endif
        std::size_t cnt = 0;
        #pragma omp parallel for reduction(+:cnt)
        for (int i = begin; i < end - begin; ++i) {
            if (functor(v[i]))
                ++cnt;
        }
        return cnt;
    }

    template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    void exclusive_scan(s_vec<T> &v_input, s_vec<T> &v_output, std::size_t const in_begin, std::size_t const in_end,
            std::size_t const out_begin, T const init) noexcept {
#ifdef DEBUG_ON
        assert(in_begin <= in_end);
#endif
        s_vec<T> v_t_size;
        s_vec<T> v_t_offset;
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            #pragma omp single
            {
                v_t_size.resize(omp_get_num_threads(), 0);
            }
            int t_size = magma_util::get_block_size(tid, in_end - in_begin, v_t_size.size());
            int t_offset = magma_util::get_block_offset(tid, in_end - in_begin, v_t_size.size());
            T size_sum = 0;
            for (int i = t_offset; i < t_offset + t_size; ++i) {
                size_sum += v_input[i + in_begin];
            }
            v_t_size[tid] = size_sum;
            #pragma omp barrier
            #pragma omp single
            {
                v_t_offset = v_t_size;
                v_t_offset[0] = 0;
                for (int i = 1; i < v_t_offset.size(); ++i) {
                    v_t_offset[i] = v_t_offset[i - 1] + v_t_size[i - 1];
                }
            }
            v_output[t_offset] = v_t_offset[tid];
            for (int i = t_offset + 1; i < t_offset + t_size; ++i) {
                v_output[out_begin + i] = v_input[i + in_begin - 1] + v_output[out_begin + i - 1];
            }
        }
    }

    template <typename T, typename F, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    void copy_if(s_vec<T> &v_input, s_vec<T> &v_output, std::size_t const in_begin, std::size_t const in_end,
            std::size_t const out_begin, F const &functor) noexcept {
#ifdef DEBUG_ON
        assert(in_begin <= in_end);
#endif
        s_vec<int> v_copy_size(v_input.size() - in_begin, 0);
        s_vec<int> v_copy_offset(v_copy_size.size());
        #pragma omp parallel for
        for (int i = 0; i < v_copy_size.size(); ++i) {
            if (functor(v_input[i+in_begin])) {
                v_copy_size[i] = 1;
            }
        }
        auto out_size = count_if(v_copy_size, 0, v_copy_size.size(), [](T const &v) -> bool {
            if (v == 1)
                return true;
            return false;
        });
        v_output.resize(out_size);
        exclusive_scan(v_copy_size, v_copy_offset, 0, v_copy_size.size(), 0, static_cast<T>(in_begin));
        #pragma omp parallel for
        for (int i = 0; i < v_copy_size.size(); ++i) {
            if (v_copy_size[i] == 1) {
                v_output[v_copy_offset[i]] = v_input[i+in_begin];
            }
        }
    }

    template <typename T, typename F, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    void for_each(s_vec<T> &v, std::size_t const begin, std::size_t const end, F const &functor) {
#ifdef DEBUG_ON
        assert(begin <= end);
#endif
        #pragma omp parallel for
        for (std::size_t i = begin; i < end - begin; ++i) {
            functor(static_cast<T>(i));
        }
    }

    template <typename T, typename F, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    std::pair<T, T> minmax_element(s_vec<T> &v, std::size_t const begin, std::size_t const end, F const &functor) noexcept {
#ifdef DEBUG_ON
        assert(begin <= end);
#endif
        s_vec<T> v_t_min_max;
        T min, max;
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            T min_t = 0;
            T max_t = 0;
            #pragma omp single
            {
                v_t_min_max.resize(omp_get_num_threads()*2, 0);
            }
            #pragma omp for
            // TODO use begin and end
            for (int i = 1; i < v.size(); ++i) {
                if (functor(v[i], v[min_t])) {
                    min_t = i;
                } else if (functor(v[max_t], v[i])) {
                    max_t = i;
                }
            }
            v_t_min_max[tid*2] = min_t;
            v_t_min_max[tid*2+1] = max_t;
        };
        std::sort(v_t_min_max.begin(), v_t_min_max.end(), functor);
        min = v_t_min_max[0];
        max = v_t_min_max[v_t_min_max.size()-1];
#ifdef EXT_DEBUG_ON
        auto minmax = std::minmax_element(std::next(v.begin(), begin), std::next(v.begin(), end), functor);
        assert(*minmax.first == v[min] && *minmax.second == v[max]);
#endif
        return std::make_pair(v[min], v[max]);
    }

    template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    T reduce(s_vec<T> &v, std::size_t const begin, std::size_t const end, T const startval) {
#ifdef DEBUG_ON
        assert(begin <= end);
        assert((end - begin) <= (v.size() - begin));
#endif
        T sum = startval;
        #pragma omp parallel for reduction(+: sum)
        for (std::size_t i = begin; i < end - begin; ++i) {
            sum += v[i];
        }
        return sum;
    }

    //TODO
    template <typename T, typename F, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    void sort(s_vec<T> &v, std::size_t const begin, std::size_t const end, F const &functor) noexcept {
#ifdef DEBUG_ON
        assert(begin <= end);
        assert((end - begin) <= (v.size() - begin));
#endif
        if (end - begin < 10000) {
            std::sort(std::next(v.begin(), begin), std::next(v.begin(), end), functor);
            return;
        }
        std::sort(std::next(v.begin(), begin), std::next(v.begin(), end), functor);
    }

    template <typename T1, typename T2, typename F, typename std::enable_if<std::is_arithmetic<T1>::value>::type* = nullptr>
    void unique(s_vec<T1> &v_input, s_vec<T2> &v_output, std::size_t const in_begin, std::size_t const in_end,
            std::size_t const out_begin, F const &functor) {
#ifdef DEBUG_ON
        assert(in_begin <= in_end);
#endif
        v_output.resize(1, 0);
        exa::copy_if(v_input, v_output, 1, v_input.size(), 1, functor);
    }

    template <typename T1, typename T2, typename F, typename std::enable_if<std::is_arithmetic<T1>::value>::type* = nullptr>
    void transform(s_vec<T1> &v_input, s_vec<T2> &v_output, std::size_t const in_begin, std::size_t const in_end,
            std::size_t const out_begin, F const &functor) noexcept {
#ifdef DEBUG_ON
        assert(in_begin <= in_end);
#endif
        #pragma omp parallel for
        for (std::size_t i = 0; i < in_end - in_begin; ++i) {
            v_output[out_begin + i] = functor(v_input[i + in_begin]);
        }
    }
}

#endif //EXAFOUNDRY_EXA_H
