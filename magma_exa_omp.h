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
#include <cmath>
#include <iostream>
#include <omp.h>
#include "magma_util.h"

template <typename T>
using d_vec = std::vector<T>;
template <typename T>
using h_vec = std::vector<T>;
template <typename T>
using d_vec = std::vector<T>;

namespace exa {

    template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    void fill(d_vec<T> &v, std::size_t const begin, std::size_t const end, T const val) noexcept {
#ifdef DEBUG_ON
        assert(begin <= end);
#endif
        #pragma omp parallel for
        for (std::size_t i = begin; i < end; ++i) {
            v[i] = val;
        }
    }

    template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    void iota(d_vec<T> &v, std::size_t const begin, std::size_t const end, std::size_t const startval) noexcept {
#ifdef DEBUG_ON
        assert(begin <= end);
#endif
        #pragma omp parallel for
        for (std::size_t i = begin; i < end; ++i) {
            v[i] = startval + i - begin;
        }
    }

    template <typename T, typename F, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    std::size_t count_if(d_vec<T> &v, std::size_t const begin, std::size_t const end, F const &functor) {
#ifdef DEBUG_ON
        assert(begin <= end);
#endif
        std::size_t cnt = 0;
        #pragma omp parallel for reduction(+:cnt)
        for (int i = begin; i < end; ++i) {
            if (functor(v[i]))
                ++cnt;
        }
        return cnt;
    }

    template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    void exclusive_scan(d_vec<T> &v_input, d_vec<T> &v_output, std::size_t const in_begin, std::size_t const in_end,
            std::size_t const out_begin, T const init) noexcept {
#ifdef DEBUG_ON
        assert(in_begin <= in_end);
#endif
        d_vec<T> v_t_size;
        d_vec<T> v_t_offset;
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            #pragma omp single
            {
                v_t_size.resize(omp_get_num_threads(), 0);
            }
            int t_size = magma_util::get_block_size(tid, static_cast<int>(in_end - in_begin), static_cast<int>(v_t_size.size()));
            int t_offset = magma_util::get_block_offset(tid, static_cast<int>(in_end - in_begin), static_cast<int>(v_t_size.size()));
            T size_sum = 0;
            for (int i = t_offset; i < t_offset + t_size; ++i) {
                size_sum += v_input[i + in_begin];
            }
            v_t_size[tid] = size_sum;
            #pragma omp barrier
            #pragma omp single
            {
                v_t_offset = v_t_size;
                v_t_offset[0] = init;
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
    void copy_if(d_vec<T> &v_input, d_vec<T> &v_output, std::size_t const in_begin, std::size_t const in_end,
            std::size_t const out_begin, F const &functor) noexcept {
#ifdef DEBUG_ON
        assert(in_begin <= in_end);
#endif
        d_vec<T> v_copy_size(v_input.size() - in_begin, 0);
        d_vec<T> v_copy_offset(v_copy_size.size());
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
        for (std::size_t i = 0; i < v_copy_size.size(); ++i) {
            if (v_copy_size[i] == 1) {
                v_output[v_copy_offset[i]] = v_input[i+in_begin];
            }
        }
    }

    template <typename T, typename F, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    void for_each(d_vec<T> &v, std::size_t const begin, std::size_t const end, F const &functor) {
#ifdef DEBUG_ON
        assert(begin <= end);
#endif
        #pragma omp parallel for
        for (std::size_t i = begin; i < end; ++i) {
            functor(v[i]);
        }
    }

    template <typename T, typename F, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    std::pair<T, T> minmax_element(d_vec<T> &v, std::size_t const begin, std::size_t const end, F const &functor) noexcept {
#ifdef DEBUG_ON
        assert(begin <= end);
#endif
        d_vec<T> v_t_min_max;
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            #pragma omp single
            {
                v_t_min_max.resize(omp_get_num_threads()*2, 0);
            }
            int size = magma_util::get_block_size(tid, (int)(end - begin), static_cast<int>(v_t_min_max.size() / 2));
            int offset = magma_util::get_block_offset(tid, (int)(end - begin), static_cast<int>(v_t_min_max.size() / 2));
            auto pair = std::minmax_element(std::next(v.begin(), offset), std::next(v.begin(), offset + size),
                    functor);
            v_t_min_max[tid * 2] = *pair.first;
            v_t_min_max[tid * 2 + 1] = *pair.second;
        };
        std::sort(v_t_min_max.begin(), v_t_min_max.end(), functor);
        return std::make_pair(v_t_min_max[0], v_t_min_max[v_t_min_max.size()-1]);
    }

    template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    T reduce(d_vec<T> &v, std::size_t const begin, std::size_t const end, T const startval) {
#ifdef DEBUG_ON
        assert(begin <= end);
        assert((end - begin) <= (v.size() - begin));
#endif
        T sum = startval;
        #pragma omp parallel for reduction(+: sum)
        for (std::size_t i = begin; i < end; ++i) {
            sum += v[i];
        }
        return sum;
    }

    template <typename T, typename F, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    void sort(d_vec<T> &v, std::size_t const begin, std::size_t const end, F const &functor) noexcept {
#ifdef DEBUG_ON
        assert(begin <= end);
        assert((end - begin) <= (v.size() - begin));
#endif
        // TODO improve heuristic
        if (end - begin < 100000) {
            std::sort(std::next(v.begin(), begin), std::next(v.begin(), end), functor);
            return;
        }
        d_vec<T> v_tmp(v.size(), -1);
        d_vec<T> v_samples;
        d_vec<int> v_bucket_size;
        d_vec<int> v_bucket_offset;
        d_vec<int> v_par_bucket_size;
        d_vec<int> v_par_bucket_offset;
        // optimize to only use needed space
        d_vec<int> v_bucket_index(v.size());
        int n_thread = 0, n_bucket = 0;
        #pragma omp parallel
        {
            #pragma omp single
            {
                n_thread = omp_get_num_threads();
                int n_samples = n_thread * log10f(v.size());
                v_samples.reserve(n_samples);
                for (int i = 0; i < n_samples; ++i) {
                    v_samples.push_back(v[((v.size() - 1) / n_samples) * i]);
                }
//                std::cout << "sample size: " << n_samples << std::endl;
                std::sort(v_samples.begin(), v_samples.end(), functor);
                n_bucket = v_samples.size() + 1;
                v_bucket_size.resize(n_bucket, 0);
                v_bucket_offset.resize(n_bucket);
                v_par_bucket_size.resize(n_bucket * n_thread, -1);
                v_par_bucket_offset.resize(n_bucket * n_thread);
            }
            d_vec<int> v_t_size(n_bucket, 0);
            d_vec<int> v_t_offset(n_bucket);
            int tid = omp_get_thread_num();
            int t_size = magma_util::get_block_size(tid, static_cast<int>(end - begin), n_thread);
            int t_offset = magma_util::get_block_offset(tid, static_cast<int>(end - begin), n_thread);
            for (std::size_t i = t_offset; i < t_offset + t_size; ++i) {
                v_bucket_index[i + begin] = std::lower_bound(v_samples.begin(), v_samples.end(), v[i + begin], functor)
                        - v_samples.begin();
                ++v_t_size[v_bucket_index[i + begin]];
            }
            for (int i = 0; i < n_bucket; ++i) {
                v_par_bucket_size[(i * n_thread) + tid] = v_t_size[i];
            }
            #pragma omp barrier
            for (int i = 0; i < v_bucket_size.size(); ++i) {
                #pragma omp atomic
                v_bucket_size[i] += v_t_size[i];
            }
            #pragma omp barrier
            #pragma omp single
            {
//                magma_util::print_vector("bucket sizes: ", v_bucket_size);
                exclusive_scan(v_par_bucket_size, v_par_bucket_offset, 0, v_par_bucket_size.size(), 0, 0);
                exclusive_scan(v_bucket_size, v_bucket_offset, 0, v_bucket_size.size(), 0, 0);
            }
            for (int i = 0; i < n_bucket; ++i) {
                v_t_offset[i] = v_par_bucket_offset[i * n_thread + tid];
            }
            auto v_t_offset_cpy = v_t_offset;
            for (std::size_t i = t_offset; i < t_offset + t_size; ++i) {
                v_tmp[v_t_offset[v_bucket_index[i + begin]] + begin] = v[i + begin];
                ++v_t_offset[v_bucket_index[i + begin]];
            }
            #pragma omp barrier
            #pragma omp for schedule(dynamic)
            for (int i = 0; i < v_bucket_size.size(); ++i) {
                std::sort(std::next(v_tmp.begin(), v_bucket_offset[i]), std::next(v_tmp.begin(),
                        v_bucket_offset[i] + v_bucket_size[i]), functor);
            }
        }
        v.clear();
        v.insert(v.end(), std::make_move_iterator(v_tmp.begin()), std::make_move_iterator(v_tmp.end()));
    }

    template <typename T1, typename T2, typename F, typename std::enable_if<std::is_arithmetic<T1>::value>::type* = nullptr>
    void unique(d_vec<T1> &v_input, d_vec<T2> &v_output, std::size_t const in_begin, std::size_t const in_end,
            std::size_t const out_begin, F const &functor) {
#ifdef DEBUG_ON
        assert(in_begin <= in_end);
#endif
        v_output.resize(1, 0);
        exa::copy_if(v_input, v_output, 1, v_input.size(), 1, functor);
    }

    template <typename T1, typename T2, typename F, typename std::enable_if<std::is_arithmetic<T1>::value>::type* = nullptr>
    void transform(d_vec<T1> &v_input, d_vec<T2> &v_output, std::size_t const in_begin, std::size_t const in_end,
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
