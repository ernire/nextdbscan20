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
#include <iostream>
#include <omp.h>

template <typename T>
using s_vec = std::vector<T>;

namespace exa {
    //TODO
    template <typename T, typename F, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    void copy_if(s_vec<T> &v_input, s_vec<T> &v_output, std::size_t const in_begin, std::size_t const in_end,
            std::size_t const out_begin, F const &functor) {
#ifdef DEBUG_ON
        assert(in_begin <= in_end);
#endif
        if (v_output.size() < v_input.size() + out_begin) {
            v_output.resize(out_begin + in_end - in_begin);
        }
        auto it = std::copy_if(std::next(v_input.begin(), in_begin),
                std::next(v_input.begin(), in_end), std::next(v_output.begin(), out_begin), functor);
        v_output.resize(std::distance(v_output.begin(), it));
    }

    // TODO
    template <typename T, typename F, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    T count_if(s_vec<T> &v, std::size_t const begin, std::size_t const end, F const &functor) {
#ifdef DEBUG_ON
        assert(begin <= end);
#endif
        return std::count_if(std::next(v.begin(), begin),std::next(v.begin(), end), functor);
    }

    //TODO
    template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    void fill(s_vec<T> &v, std::size_t const begin, std::size_t const end, T const val) noexcept {
#ifdef DEBUG_ON
        assert(begin <= end);
#endif
        std::fill(std::next(v.begin(), begin), std::next(v.begin(), end), val);
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

    //TODO
    template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    void exclusive_scan(s_vec<T> &v_input, s_vec<T> &v_output, std::size_t const in_begin, std::size_t const in_end,
            std::size_t const out_begin, T const init) noexcept {
#ifdef DEBUG_ON
        assert(in_begin <= in_end);
#endif
        std::exclusive_scan(std::next(v_input.begin(), in_begin), std::next(v_input.begin(), in_end),
                std::next(v_output.begin(), out_begin), init);
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
    std::pair<T, T> minmax_element(s_vec<T> &v, std::size_t const begin, std::size_t const end, F const &functor) noexcept {
#ifdef DEBUG_ON
        assert(begin <= end);
#endif
        T min = v[begin], max = v[begin];
        s_vec<T> v_t_min, v_t_max;
        std::cout << "total threads: " << omp_get_num_threads() << std::endl;

        #pragma omp parallel
        {
            int n_threads = omp_get_num_threads();
            int tid = omp_get_thread_num();
            #pragma omp single
            {
                v_t_min.resize(n_threads);
                v_t_max.resize(n_threads);
            }
            T t_min = begin, t_max = begin;
            #pragma omp for nowait
            for (int i = begin + 1; i < end - begin; ++i) {
                if (functor(v[i], v[t_min])) {
                    t_min = i;
                } else if (functor(v[t_max], v[i])) {
                    t_max = i;
                }
            }
            v_t_min[tid] = v[t_min];
            v_t_max[tid] = v[t_max];
//            #pragma omp barrier
//            #pragma omp for reduction(max: max) reduction(min: min)
//            for (int t = 0; t < v_t_min.size(); ++t) {
//                if (v_t_min[t])
//            }
        }
//        T min = v_t_min[0];
//        T max = v_t_max[0];

//        return std::make_pair(v[min], v[max]);

        return std::make_pair(*std::min, max);
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
        std::size_t j = out_begin;
        #pragma omp parallel for
        for (std::size_t i = in_begin; i < in_end - in_begin; ++i) {
            v_output[j + i - in_begin] = functor(v_input[i]);
        }
    }
}

#endif //EXAFOUNDRY_EXA_H
