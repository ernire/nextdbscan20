//
// Created by Ernir Erlingsson on 16.8.2020.
//

#ifndef NEXTDBSCAN20_MAGMA_UTIL_H
#define NEXTDBSCAN20_MAGMA_UTIL_H

#include <iostream>
#include <random>
#include <vector>
#include <chrono>
#include <functional>

/*
template<typename T>
using random_distribution = std::conditional_t<std::is_integral<T>::value,
        std::uniform_int_distribution<T>,
        std::conditional_t<std::is_floating_point<T>::value,
                std::uniform_real_distribution<T>,
                void>
>;
*/

namespace magma_util {

    template<class T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    T get_block_size(T block_index, T number_of_samples, T number_of_blocks) noexcept {
        T block = (number_of_samples / number_of_blocks);
        T reserve = number_of_samples % number_of_blocks;
        //    Some processes will need one more sample if the data size does not fit completely
        if (reserve > 0 && block_index < reserve) {
            return block + 1;
        }
        return block;
    }

    template<class T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    T get_block_offset(T block_index, T number_of_samples, T number_of_blocks) noexcept {
        T offset = 0;
        for (T i = 0; i < block_index; i++) {
            offset += get_block_size(i, number_of_samples, number_of_blocks);
        }
        return offset;
    }

    template<class F>
    long long measure_duration(std::string const &name, bool const is_verbose, F const &functor) noexcept {
        if (is_verbose) {
            std::cout << name << std::flush;
        }
        auto start_timestamp = std::chrono::high_resolution_clock::now();
        functor();
        auto end_timestamp = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_timestamp - start_timestamp).count();
        if (is_verbose) {
            std::cout << duration << " milliseconds" << std::endl;
        }
        return duration;
    }

    template<class T>
    void print_v(const std::string &name, T *v, std::size_t size) noexcept {
        std::cout << name;
        for (int i = 0; i < size; ++i) {
            std::cout << v[i] << " ";
        }
        std::cout << std::endl;
    }

    /*
    template<class T, typename std::enable_if<std::is_arithmetic<T>::value>::type * = nullptr>
    void random_vector(std::vector<T> &vec, const size_t pool_size) noexcept {
        // TODO not constant seed value
        std::default_random_engine generator(12345);
        random_distribution<T> rnd_dist(0, pool_size);
        auto rnd_gen = std::bind(rnd_dist, generator);
        for (auto &val : vec) {
            val = rnd_gen();
        }
    }
     */

}

#endif //NEXTDBSCAN20_MAGMA_UTIL_H
