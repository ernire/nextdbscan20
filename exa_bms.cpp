//
// Created by Ernir Erlingsson on 18.9.2020.
//
#include <iostream>
#ifdef OMP_ON
#include <omp.h>
#endif
#include "magma_meta.h"

template<typename T>
using random_distribution = std::conditional_t<std::is_integral<T>::value,
        std::uniform_int_distribution<T>,
        std::conditional_t<std::is_floating_point<T>::value,
                std::uniform_real_distribution<T>,
                void>
>;

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

void count_if_bm(s_vec<long long> &v_input, s_vec<long long> &v_time, int const n_iter) {
    for (long long i = 0; i < n_iter; ++i) {
#ifdef OMP_ON
        omp_set_num_threads(static_cast<int>(powf(2, i)));
#endif
        v_time[i] = magma_util::measure_duration("", false, [&]() -> void {
            auto cnt = exa::count_if(v_input, 0, v_input.size(), [](auto const &v) -> bool {
                return v % 2 == 0;
            });
        });
    }
    magma_util::print_v("Exa count_if Times: ", &v_time[0], v_time.size());
}

void copy_bm(s_vec<long long> &v_input, s_vec<long long> &v_output, s_vec<long long> &v_time, int const n_iter) {
    for (long long i = 0; i < n_iter; ++i) {
#ifdef OMP_ON
        omp_set_num_threads(static_cast<int>(powf(2, i)));
#endif
        v_output.resize(v_input.size());
        v_time[i] = magma_util::measure_duration("", false, [&]() -> void {
            exa::copy_if(v_input, v_output, 0, v_input.size(), 0, [](auto const &v) -> bool {
                return v % 2 == 0;
            });
        });
    }
    magma_util::print_v("Exa copy_if Times: ", &v_time[0], v_time.size());
}

void fill_bm(s_vec<long long> &v_input, s_vec<long long> &v_time, int const n_iter) {
    for (long long i = 0; i < n_iter; ++i) {
#ifdef OMP_ON
        omp_set_num_threads(static_cast<int>(powf(2, i)));
#endif
        v_time[i] = magma_util::measure_duration("", false, [&]() -> void {
            exa::fill(v_input, 0, v_input.size(), i);
        });
    }
    magma_util::print_v("Exa fill Times: ", &v_time[0], v_time.size());
}

void for_each_bm(s_vec<long long> &v_input, s_vec<long long> &v_work, s_vec<long long> &v_time, int const n_iter) {
    for (long long i = 0; i < n_iter; ++i) {
#ifdef OMP_ON
        omp_set_num_threads(static_cast<int>(powf(2, i)));
#endif
        v_time[i] = magma_util::measure_duration("", false, [&]() -> void {
            exa::for_each(v_input, 0, v_input.size(), [&](auto &v) -> void {
                long long sum = 0;
                for (auto const &v2 : v_work) {
                    sum += v2;
                }
                v = sum;
            });
        });
    }
    magma_util::print_v("Exa for_each Times: ", &v_time[0], v_time.size());
}

void iota_bm(s_vec<long long> &v_input, s_vec<long long> &v_time, int const n_iter) {
    for (long long i = 0; i < n_iter; ++i) {
#ifdef OMP_ON
        omp_set_num_threads(static_cast<int>(powf(2, i)));
#endif
        v_time[i] = magma_util::measure_duration("", false, [&]() -> void {
            exa::iota(v_input, 0, v_input.size(), i);
        });
    }
    magma_util::print_v("Exa iota Times: ", &v_time[0], v_time.size());
}

void exclusive_scan_bm(s_vec<long long> &v_input, s_vec<long long> &v_output, s_vec<long long> &v_time, int const n_iter) {
    for (long long i = 0; i < n_iter; ++i) {
#ifdef OMP_ON
        omp_set_num_threads(static_cast<int>(powf(2, i)));
#endif
        exa::fill(v_input, 0, v_input.size(), static_cast<long long>(1));
        v_time[i] = magma_util::measure_duration("", false, [&]() -> void {
            exa::exclusive_scan(v_input, v_output, 0, v_input.size(), 0, i);
        });
    }
    magma_util::print_v("Exa iota Times: ", &v_time[0], v_time.size());
}

void minmax_element_bm(s_vec<long long> &v_input, s_vec<long long> &v_time, int const n_iter) {
    for (long long i = 0; i < n_iter; ++i) {
#ifdef OMP_ON
        omp_set_num_threads(static_cast<int>(powf(2, i)));
#endif
        v_time[i] = magma_util::measure_duration("", false, [&]() -> void {
            exa::minmax_element(v_input, 0, v_input.size(), [](auto const v1, auto const v2) -> bool {
                return v1 < v2;
            });
        });
    }
    magma_util::print_v("Exa minmax_element Times: ", &v_time[0], v_time.size());
}

void reduce_bm(s_vec<long long> &v_input, s_vec<long long> &v_time, int const n_iter) {
    for (long long i = 0; i < n_iter; ++i) {
#ifdef OMP_ON
        omp_set_num_threads(static_cast<int>(powf(2, i)));
#endif
        v_time[i] = magma_util::measure_duration("", false, [&]() -> void {
            exa::reduce(v_input, 0, v_input.size(), static_cast<long long>(0));
        });
    }
    magma_util::print_v("Exa reduce Times: ", &v_time[0], v_time.size());
}

void sort_bm(s_vec<long long> &v_input, s_vec<long long> &v_time, int const n_iter) {
    for (long long i = 0; i < n_iter; ++i) {
#ifdef OMP_ON
        omp_set_num_threads(static_cast<int>(powf(2, i)));
#endif
        auto v_copy = v_input;
        v_time[i] = magma_util::measure_duration("", false, [&]() -> void {
            exa::sort(v_copy, 0, v_copy.size(), [](auto const &v1, auto const &v2) -> bool {
                return v1 < v2;
            });
        });
    }
    magma_util::print_v("Exa sort Times: ", &v_time[0], v_time.size());
}

void transform_bm(s_vec<long long> &v_input, s_vec<long long> &v_time, int const n_iter) {
    for (long long i = 0; i < n_iter; ++i) {
#ifdef OMP_ON
        omp_set_num_threads(static_cast<int>(powf(2, i)));
#endif
        v_time[i] = magma_util::measure_duration("", false, [&]() -> void {
            exa::transform(v_input, v_input, 0, v_input.size(), 0, [](auto const &v) -> long long {
                return -v;
            });
        });
    }
    magma_util::print_v("Exa count_if Times: ", &v_time[0], v_time.size());
}

void unique_bm(s_vec<long long> &v_input, s_vec<long long> &v_time, int const n_iter) {
    s_vec<long long> v_iota(v_input.size());
    exa::iota(v_iota, 0, v_iota.size(), 0);
    for (long long i = 0; i < n_iter; ++i) {
#ifdef OMP_ON
        omp_set_num_threads(static_cast<int>(powf(2, i)));
#endif
        auto v_copy = v_input;
        v_time[i] = magma_util::measure_duration("", false, [&]() -> void {
            exa::unique(v_iota, v_copy, 0, v_iota.size(), 0, [&](auto const &i) -> bool {
                return v_input[i-1] != v_input[i];
            });
        });
    }
    magma_util::print_v("Exa unique Times: ", &v_time[0], v_time.size());
}

int main(int argc, char **argv) {
    std::cout << "Starting Exa Benchmark Suite" << std::endl;

    int const INT32 = INT32_MAX;
    int const MB100 = 100000000;
    s_vec<long long> v_input(INT32);
    int n_iter = 1;
    long long cnt = 0;
#ifdef OMP_ON
    #pragma omp parallel
    {
        // + 1 for the inclusion of hyperthreading
        n_iter = log2(omp_get_num_threads());
    }
#endif
    std::cout << "iterations: " << n_iter << std::endl;
    s_vec<long long> v_time(n_iter);

    // iota
    magma_util::measure_duration("std iota: ", true, [&]() -> void {
        std::iota(v_input.begin(), v_input.end(), 100);
    });
    iota_bm(v_input, v_time, n_iter);
    std::cout << std::endl;

    // transform
    magma_util::measure_duration("std transform: ", true, [&]() -> void {
        std::transform(v_input.begin(), v_input.end(), v_input.begin(), [](auto &v) -> long long {
            return -v;
        });
    });
    transform_bm(v_input, v_time, n_iter);
    std::cout << std::endl;

    // count_if
    magma_util::measure_duration("std count_if: ", true, [&]() -> void {
        cnt = std::count_if(v_input.begin(), v_input.end(), [](auto const &v) -> bool {
            return v % 2 == 0;
        });
    });
    std::cout << "count result: " << cnt << std::endl;
    count_if_bm(v_input, v_time, n_iter);
    std::cout << std::endl;

    // minmax_element
    magma_util::measure_duration("std minmax_element: ", true, [&]() -> void {
        std::minmax_element(v_input.begin(), v_input.end());
    });
    minmax_element_bm(v_input, v_time, n_iter);
    std::cout << std::endl;

    // fill
    magma_util::measure_duration("std fill: ", true, [&]() -> void {
        std::fill(v_input.begin(), v_input.end(), 100);
    });
    fill_bm(v_input, v_time, n_iter);
    std::cout << std::endl;

    // copy_if
    v_input.resize(MB100);
    std::iota(v_input.begin(), v_input.end(), 0);
    s_vec<long long> v_copy(v_input.size());
    magma_util::measure_duration("std copy_if: ", true, [&]() -> void {
        auto it = std::copy_if(v_input.begin(), v_input.end(), v_copy.begin(), [](auto const &v) -> bool {
            return v % 2 == 0;
        });
        v_copy.resize(std::distance(v_copy.begin(),it));
    });
    std::cout << "copy size: " << v_copy.size() << std::endl;
    copy_bm(v_input, v_copy, v_time, n_iter);
    std::cout << std::endl;

    // reduce
//    magma_util::measure_duration("std reduce: ", true, [&]() -> void {
//        cnt = std::reduce(v_input.begin(), v_input.end(), 0);
//    });
    std::cout << "reduce cnt: " << cnt << std::endl;
    reduce_bm(v_input, v_time, n_iter);
    std::cout << std::endl;

    // for_each
    v_copy.resize(100);
    magma_util::measure_duration("std for_each: ", true, [&]() -> void {
        std::for_each(v_input.begin(), v_input.end(), [&](auto &v) -> void {
            long long sum = 0;
            for (auto const &v2 : v_copy) {
                sum += v2;
            }
            v = sum;
        });
    });
    for_each_bm(v_input, v_copy, v_time, n_iter);
    std::cout << std::endl;


    // sort
    v_input.resize(MB100);
    s_vec<long long> v_output(MB100);
    // exclusive_scan
    exclusive_scan_bm(v_input, v_output, v_time, n_iter);


    random_vector(v_input, v_input.size() / 2);
    v_copy = v_input;
    magma_util::measure_duration("std sort: ", true, [&]() -> void {
        std::sort(v_copy.begin(), v_copy.end());
    });
    sort_bm(v_input, v_time, n_iter);
    std::cout << std::endl;

    // unique
    v_input = v_copy;
    magma_util::measure_duration("std unique: ", true, [&]() -> void {
        auto last = std::unique(v_copy.begin(), v_copy.end());
        v_copy.erase(last, v_copy.end());
    });
    unique_bm(v_input, v_time, n_iter);

}