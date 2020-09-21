//
// Created by Ernir Erlingsson on 18.9.2020.
//

#ifndef NEXTDBSCAN20_MAGMA_EXA_CU_CUH
#define NEXTDBSCAN20_MAGMA_EXA_CU_CUH

#include <cassert>
#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>

template <typename T>
using h_vec = thrust::host_vector<T>;
template <typename T>
using d_vec = thrust::device_vector<T>;

namespace exa {
    /*
    template<typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type * = nullptr>
    void fill(s_vec<T> &v, std::size_t const begin, std::size_t const end, T const val) noexcept {
#ifdef DEBUG_ON
        assert(begin <= end);
#endif
        thrust::fill(std::next(v.begin(), begin), std::next(v.begin(), end), val);
    }
    */
//    void iota(d_vec<int> &v, std::size_t const begin, std::size_t const end, int const startval) noexcept;

//    template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
//    void iota(d_vec<T> &v, std::size_t const begin, std::size_t const end, std::size_t const startval) noexcept {
//        thrust::sequence(v.begin(), v.begin());
//    }
    void iota(d_vec<int> &v, std::size_t const begin, std::size_t const end, int const startval) noexcept;

    void iota(d_vec<long long> &v, std::size_t const begin, std::size_t const end, long long const startval) noexcept;

//#ifdef DEBUG_ON
//        assert(begin <= end);
//#endif
//        thrust::sequence(std::next(v.begin(), begin), std::next(v.begin(), end), startval);
//        thrust::sequence(v.begin()+begin, v.begin()+end, startval);
//    }
    /*
    template<typename T, typename F, typename std::enable_if<std::is_arithmetic<T>::value>::type * = nullptr>
    std::size_t count_if(s_vec<T> &v, std::size_t const begin, std::size_t const end, F const &functor) {
#ifdef DEBUG_ON
        assert(begin <= end);
#endif
//        return thrust::count_if(std::next(v.begin(), begin),std::next(v.begin(), end), functor);
        return 0;
    }

    template<typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type * = nullptr>
    void exclusive_scan(s_vec<T> &v_input, s_vec<T> &v_output, std::size_t const in_begin, std::size_t const in_end,
            std::size_t const out_begin, T const init) noexcept {
#ifdef DEBUG_ON
        assert(in_begin <= in_end);
#endif
        thrust::exclusive_scan(std::next(v_input.begin(), in_begin), std::next(v_input.begin(), in_end),
                std::next(v_output.begin(), out_begin), init);
    }

    template<typename T, typename F, typename std::enable_if<std::is_arithmetic<T>::value>::type * = nullptr>
    void copy_if(s_vec<T> &v_input, s_vec<T> &v_output, std::size_t const in_begin, std::size_t const in_end,
            std::size_t const out_begin, F const &functor) noexcept {
#ifdef DEBUG_ON
        assert(in_begin <= in_end);
#endif
        if (v_output.size() < v_input.size() + out_begin) {
            v_output.resize(out_begin + in_end - in_begin);
        }
//        auto it = thrust::copy_if(std::next(v_input.begin(), in_begin),
//                std::next(v_input.begin(), in_end), std::next(v_output.begin(), out_begin), functor);
//        v_output.resize(std::distance(v_output.begin(), it));
    }

    template<typename T, typename F, typename std::enable_if<std::is_arithmetic<T>::value>::type * = nullptr>
    void for_each(s_vec<T> &v, std::size_t const begin, std::size_t const end, F const &functor) {
#ifdef DEBUG_ON
        assert(begin <= end);
#endif
//        thrust::for_each(std::next(v.begin(), begin), std::next(v.begin(), end), functor);
    }

    template<typename T, typename F, typename std::enable_if<std::is_arithmetic<T>::value>::type * = nullptr>
    std::pair<T, T>
    minmax_element(s_vec<T> &v, std::size_t const begin, std::size_t const end, F const &functor) noexcept {
#ifdef DEBUG_ON
        assert(begin <= end);
#endif
//        auto minmax = thrust::minmax_element(std::next(v.begin(), begin), std::next(v.begin(), end), functor);
//        return std::make_pair(*minmax.first, *minmax.second);
        return std::make_pair((T)0, (T)0);
    }

    template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    T reduce(s_vec<T> &v, std::size_t const begin, std::size_t const end, T const startval) {
#ifdef DEBUG_ON
        assert(begin <= end);
        assert((end - begin) <= (v.size() - begin));
#endif
        return thrust::reduce(std::next(v.begin(), begin), std::next(v.begin(), end), startval);
    }

    template<typename T, typename F, typename std::enable_if<std::is_arithmetic<T>::value>::type * = nullptr>
    void sort(s_vec<T> &v, std::size_t const begin, std::size_t const end, F const &functor) noexcept {
#ifdef DEBUG_ON
        assert(begin <= end);
        assert((end - begin) <= (v.size() - begin));
#endif
//        thrust::sort(std::next(v.begin(), begin), std::next(v.begin(), end), functor);
    }

    template<typename T1, typename T2, typename F, typename std::enable_if<std::is_arithmetic<T1>::value>::type * = nullptr>
    void unique(s_vec<T1> &v_input, s_vec<T2> &v_output, std::size_t const in_begin, std::size_t const in_end,
            std::size_t const out_begin, F const &functor) {
#ifdef DEBUG_ON
        assert(in_begin <= in_end);
#endif
//        thrust::unique(std::next(v_input.begin(), in_begin), std::next(v_input.begin(), in_end),
//                std::next(v_output.begin(), out_begin), functor);
        v_output.resize(1, 0);
        exa::copy_if(v_input, v_output, 1, v_input.size(), 1, functor);
    }

    template <typename T1, typename T2, typename F, typename std::enable_if<std::is_arithmetic<T1>::value>::type* = nullptr>
    void transform(s_vec<T1> &v_input, s_vec<T2> &v_output, std::size_t const in_begin, std::size_t const in_end,
            std::size_t const out_begin, F const &functor) noexcept {
#ifdef DEBUG_ON
        assert(in_begin <= in_end);
#endif
        thrust::transform(std::next(v_input.begin(), in_begin), std::next(v_input.begin(), in_end),
                std::next(v_output.begin(), out_begin), functor);
    }
     */
};
#endif //NEXTDBSCAN20_MAGMA_EXA_CU_CUH
