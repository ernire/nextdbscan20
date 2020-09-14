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

template<typename T>
using random_distribution = std::conditional_t<std::is_integral<T>::value,
        std::uniform_int_distribution<T>,
        std::conditional_t<std::is_floating_point<T>::value,
                std::uniform_real_distribution<T>,
                void>
>;

namespace magma_util {

    int get_block_size(int block_index, int number_of_samples, int number_of_blocks) noexcept;

    int get_block_offset(int block_index, int number_of_samples, int number_of_blocks) noexcept;

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
    void print_vector_n(const std::string &name, std::vector<T> &v_vec, int n) noexcept {
        std::cout << name;
        for (int i = 0; i < v_vec.size() && i < n; ++i) {
            std::cout << v_vec[i] << " ";
        }
        std::cout << std::endl;
    }

    template<class T, typename std::enable_if<std::is_arithmetic<T>::value>::type * = nullptr>
    void max_elem(const std::string &name, std::vector<T> &v_vec) noexcept {
        T max = 0;
        for (int i = 0; i < v_vec.size(); ++i) {
            if (v_vec[i] > max) {
                max = v_vec[i];
            }
        }
        std::cout << name;
        std::cout << " max value is " << max << std::endl;
//        return max;
//        std::cout << name;
//        for (int i = 0; i < v_vec.size(); ++i) {
//            std::cout << v_vec[i] << " ";
//        }
//        std::cout << std::endl;
    }

    template<class T>
    void print_vector(const std::string &name, std::vector<T> &v_vec) noexcept {
        std::cout << name;
        for (int i = 0; i < v_vec.size(); ++i) {
            std::cout << v_vec[i] << " ";
        }
        std::cout << std::endl;
    }

    template<class T, typename std::enable_if<std::is_arithmetic<T>::value>::type * = nullptr>
    void random_vector(std::vector<T> &vec, const size_t pool_size) noexcept {
//        std::default_random_engine generator(std::random_device{}());
        // TODO not constant seed value
        std::default_random_engine generator(12345);
        random_distribution<T> rnd_dist(0, pool_size);
        auto rnd_gen = std::bind(rnd_dist, generator);
        for (auto &val : vec) {
            val = rnd_gen();
        }
    }
}

#endif //NEXTDBSCAN20_MAGMA_UTIL_H
