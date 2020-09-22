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

namespace exa {
    template <typename T, typename F, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    void copy_if(d_vec<T> &v_input, d_vec<T> &v_output, std::size_t const in_begin, std::size_t const in_end,
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

    template <typename T, typename F, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    T count_if(d_vec<T> &v, std::size_t const begin, std::size_t const end, F const &functor) {
#ifdef DEBUG_ON
        assert(begin <= end);
#endif
        return std::count_if(std::next(v.begin(), begin),std::next(v.begin(), end), functor);
    }

    template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    void fill(d_vec<T> &v, std::size_t const begin, std::size_t const end, T const val) noexcept {
#ifdef DEBUG_ON
        assert(begin <= end);
#endif
        std::fill(std::next(v.begin(), begin), std::next(v.begin(), end), val);
    }

    template <typename T, typename F, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    void for_each(d_vec<T> &v, std::size_t const begin, std::size_t const end, F const &functor) {
#ifdef DEBUG_ON
        assert(begin <= end);
#endif
        std::for_each(std::next(v.begin(), begin), std::next(v.begin(), end), functor);
    }

    template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    void exclusive_scan(d_vec<T> &v_input, d_vec<T> &v_output, std::size_t const in_begin, std::size_t const in_end,
            std::size_t const out_begin, T const init) noexcept {
#ifdef DEBUG_ON
        assert(in_begin <= in_end);
#endif
        std::exclusive_scan(std::next(v_input.begin(), in_begin), std::next(v_input.begin(), in_end),
                std::next(v_output.begin(), out_begin), init);
    }

    template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    void iota(d_vec<T> &v, std::size_t const begin, std::size_t const end, int const startval) noexcept {
#ifdef DEBUG_ON
        assert(begin <= end);
#endif
        std::iota(std::next(v.begin(), begin), std::next(v.begin(), end), startval);
    }

    template <typename T, typename F, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    std::pair<T, T> minmax_element(d_vec<T> &v, std::size_t const begin, std::size_t const end, F const &functor) noexcept {
#ifdef DEBUG_ON
        assert(begin <= end);
#endif
        auto minmax = std::minmax_element(std::next(v.begin(), begin), std::next(v.begin(), end), functor);
        return std::make_pair(*minmax.first, *minmax.second);
//        std::iota(std::next(v.begin(), begin), std::next(v.begin(), end), startval);
    }

    template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    T reduce(d_vec<T> &v, std::size_t const begin, std::size_t const end, T const startval) {
#ifdef DEBUG_ON
        assert(begin <= end);
        assert((end - begin) <= (v.size() - begin));
#endif
        return std::reduce(std::next(v.begin(), begin), std::next(v.begin(), end), startval);
    }

    template <typename T, typename F, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    void sort(d_vec<T> &v, std::size_t const begin, std::size_t const end, F const &functor) noexcept {
#ifdef DEBUG_ON
        assert(begin <= end);
        assert((end - begin) <= (v.size() - begin));
#endif
        std::sort(std::next(v.begin(), begin), std::next(v.begin(), end), functor);
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
        std::transform(std::next(v_input.begin(), in_begin),
                std::next(v_input.begin(), in_end), std::next(v_output.begin(), out_begin), functor);
    }
}

#endif //EXAFOUNDRY_EXA_H
