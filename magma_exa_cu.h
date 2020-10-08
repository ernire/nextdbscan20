//
// Created by Ernir Erlingsson on 18.9.2020.
//

#ifndef NEXTDBSCAN20_MAGMA_EXA_CU_H
#define NEXTDBSCAN20_MAGMA_EXA_CU_H

#include <cassert>
#include <iostream>
#include <thrust/binary_search.h>
#include <thrust/functional.h>

namespace exa {

    template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    void fill(d_vec<T> &v, std::size_t const begin, std::size_t const end, T const val) noexcept {
#ifdef DEBUG_ON
        assert(begin <= end);
#endif

    }

    template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    void iota(d_vec<T> &v, std::size_t const begin, std::size_t const end, std::size_t const startval) noexcept {
#ifdef DEBUG_ON
        assert(begin <= end);
#endif
        thrust::sequence(v.begin() + begin, v.begin() + end, startval);
    }

    template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    T reduce(d_vec<T> &v, std::size_t const begin, std::size_t const end, T const startval) noexcept {
#ifdef DEBUG_ON
        assert(begin <= end);
        assert((end - begin) <= (v.size() - begin));
#endif
        return thrust::reduce(v.begin() + begin, v.begin() + end, startval);
    }

    template <typename T, typename F, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    std::size_t count_if(d_vec<T> &v, std::size_t const begin, std::size_t const end, F const &functor) noexcept {
#ifdef DEBUG_ON
        assert(begin <= end);
#endif
        return thrust::count_if(v.begin() + begin, v.begin() + end, functor);
    }

    template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    void exclusive_scan(d_vec<T> &v_input, d_vec<T> &v_output, std::size_t const in_begin, std::size_t const in_end,
            std::size_t const out_begin, T const init) noexcept {
#ifdef DEBUG_ON
        assert(in_begin <= in_end);
#endif
        thrust::exclusive_scan(v_input.begin() + in_begin, v_input.begin() + in_end, v_output.begin() + out_begin, init);
    }

    template <typename T, typename F, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    void copy_if(d_vec<T> &v_input, std::size_t const in_begin, std::size_t const in_end, d_vec<T> &v_output,
            std::size_t const out_begin, F const &functor) noexcept {
#ifdef DEBUG_ON
        assert(in_begin <= in_end);
#endif
        auto it = thrust::copy_if(v_input.begin() + in_begin, v_input.begin() + in_end, v_output.begin() + out_begin,
                functor);
        v_output.resize(thrust::distance(v_output.begin(), it));
    }

    template <typename F>
    void for_each(std::size_t const begin, std::size_t const end, F const &functor) noexcept {
#ifdef DEBUG_ON
        assert(begin <= end);
#endif
        thrust::counting_iterator<int> it_cnt_begin(begin);
        thrust::counting_iterator<int> it_cnt_end = it_cnt_begin + (end - begin);
        thrust::for_each(it_cnt_begin, it_cnt_end, functor);
//        auto const lam = functor;
//        thrust::for_each(it_cnt_begin, it_cnt_end, [=]__device__(auto const &i) -> void {
//            functor(i);
//        });
    }

    template <typename F>
    void for_each_experimental(std::size_t const begin, std::size_t const end, F const &functor) noexcept {
#ifdef DEBUG_ON
        assert(begin <= end);
#endif
        thrust::counting_iterator<int> it_cnt_begin(begin);
        thrust::counting_iterator<int> it_cnt_end = it_cnt_begin + (end - begin);
        thrust::for_each(thrust::device, it_cnt_begin, it_cnt_end, [=]__device__(auto const &i) {
            functor(i);
        });
    }

    template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    void lower_bound(d_vec<T> &v_input, std::size_t const in_begin, std::size_t const in_end,
            d_vec<T> &v_value, std::size_t const value_begin, std::size_t const value_end,
            d_vec<T> &v_output, std::size_t const out_begin, int const stride) noexcept {
#ifdef DEBUG_ON
        assert(in_begin <= in_end);
        assert(value_begin <= value_end);
        assert(v_input.size() >= (in_end - in_begin));
        assert(v_output.size() >= ((value_end - value_begin - 1) * (stride + 1)) + out_begin);
#endif
        thrust::counting_iterator<int> it_cnt_begin(in_begin);
        auto it_trans_begin = thrust::make_transform_iterator(it_cnt_begin, (thrust::placeholders::_1 * (stride + 1)) + out_begin);
        auto it_perm_begin = thrust::make_permutation_iterator(v_output.begin(), it_trans_begin);
        thrust::lower_bound(v_input.begin() + in_begin, v_input.begin() + in_end, v_value.begin() + value_begin,
                v_value.begin() + value_end, it_perm_begin);
    }

    template <typename T, typename F, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    thrust::pair<T, T> minmax_element(d_vec<T> &v, std::size_t const begin, std::size_t const end, F const &functor) noexcept {
#ifdef DEBUG_ON
        assert(begin <= end);
#endif
        auto pair = thrust::minmax_element(v.begin() + begin, v.begin() + end, functor);
        return thrust::make_pair(*pair.first, *pair.second);
    }

    template <typename T, typename F, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    void sort(d_vec<T> &v, std::size_t const begin, std::size_t const end, F const &functor) noexcept {
#ifdef DEBUG_ON
        assert(begin <= end);
        assert((end - begin) <= (v.size() - begin));
#endif
        thrust::sort(v.begin() + begin, v.begin() + end, functor);
    }

    template <typename T1, typename T2, typename F, typename std::enable_if<std::is_arithmetic<T1>::value>::type* = nullptr>
    void unique(d_vec<T1> &v_input, d_vec<T2> &v_output, std::size_t const in_begin, std::size_t const in_end,
            std::size_t const out_begin, F const &functor) noexcept {
#ifdef DEBUG_ON
        assert(in_begin <= in_end);
#endif
    }

    template <typename T1, typename T2, typename F, typename std::enable_if<std::is_arithmetic<T1>::value>::type* = nullptr>
    void transform(d_vec<T1> &v_input, std::size_t const in_begin, std::size_t const in_end, d_vec<T2> &v_output,
            std::size_t const out_begin, F const &functor) noexcept {
#ifdef DEBUG_ON
        assert(in_begin <= in_end);
#endif
        thrust::transform(v_input.begin() + in_begin, v_input.begin() + in_end, v_output.begin() + out_begin, functor);
    }

};
#endif //NEXTDBSCAN20_MAGMA_EXA_CU_H
