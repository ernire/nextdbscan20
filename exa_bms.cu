//
// Created by Ernir Erlingsson on 18.9.2020.
//
#include <iostream>
#include <numeric>
#include "magma_meta.h"
#include "magma_util.h"

template<class T>
void print_vector(const std::string &name, s_vec<T> &v_vec) noexcept {
    std::cout << name;
    for (int i = 0; i < v_vec.size(); ++i) {
        std::cout << v_vec[i] << " ";
    }
    std::cout << std::endl;
}

template<class T, typename std::enable_if<std::is_arithmetic<T>::value>::type * = nullptr>
void random_vector(s_vec<T> &vec, const size_t pool_size) noexcept {
    // TODO not constant seed value
    std::default_random_engine generator(12345);
    random_distribution<T> rnd_dist(0, pool_size);
    auto rnd_gen = std::bind(rnd_dist, generator);
    for (auto &val : vec) {
        val = rnd_gen();
    }
}

void print_cuda_memory_usage() {
    size_t free_byte;
    size_t total_byte;
    auto cuda_status = cudaMemGetInfo( &free_byte, &total_byte );

    if ( cudaSuccess != cuda_status ) {
        printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status) );
        exit(1);
    }
    double free_db = (double)free_byte ;
    double total_db = (double)total_byte ;
    double used_db = total_db - free_db ;
    printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",
            used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);
}

void count_if_bm(s_vec<long long> &v_input, s_vec<long long> &v_time, int const n_iter) {
    for (long long i = 0; i < n_iter; ++i) {
#ifdef OMP_ON
        omp_set_num_threads(static_cast<int>(powf(2, i)));
#endif
        v_time[i] = magma_util::measure_duration("", false, [&]() -> void {
            auto cnt = exa::count_if(v_input, 0, v_input.size(), [] __device__ (auto const &v) -> bool {
                return v % 2 == 0;
            });
        });
    }
    print_vector("Exa count_if Times: ", v_time);
}

void copy_bm(s_vec<long long> &v_input, s_vec<long long> &v_output, s_vec<long long> &v_time, int const n_iter) {
    for (long long i = 0; i < n_iter; ++i) {
#ifdef OMP_ON
        omp_set_num_threads(static_cast<int>(powf(2, i)));
#endif
        v_output.resize(v_input.size());
        v_time[i] = magma_util::measure_duration("", false, [&]() -> void {
            exa::copy_if(v_input, v_output, 0, v_input.size(), 0,
            []__device__ (auto const &v) -> bool {
                return v % 2 == 0;
            });
        });
    }
    print_vector("Exa copy_if Times: ", v_time);
}

void fill_bm(s_vec<long long> &v_input, s_vec<long long> &v_time, int const n_iter) {
    for (long long i = 0; i < n_iter; ++i) {
#ifdef OMP_ON
        omp_set_num_threads(static_cast<int>(powf(2, i)));
#endif
        v_time[i] = magma_util::measure_duration("", false, [&] () -> void {
            exa::fill(v_input, 0, v_input.size(), i);
        });
    }
    print_vector("Exa Fill Times: ", v_time);
}

void for_each_bm(s_vec<long long> &v_input, s_vec<long long> &v_work, s_vec<long long> &v_time, int const n_iter) {
    for (long long i = 0; i < n_iter; ++i) {
#ifdef OMP_ON
        omp_set_num_threads(static_cast<int>(powf(2, i)));
#endif
        v_time[i] = magma_util::measure_duration("", false, [&]() -> void {
//            exa::for_each(v_input, 0, v_input.size(), [&]__device__(auto &v) -> void {
//                long long sum = 0;
//                for (auto const &v2 : v_work) {
//                    sum += v2;
//                }
//                v = sum;
//            });
        });
    }
    print_vector("Exa for_each Times: ", v_time);
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
    print_vector("Exa iota Times: ", v_time);
}

void minmax_element_bm(s_vec<long long> &v_input, s_vec<long long> &v_time, int const n_iter) {
    for (long long i = 0; i < n_iter; ++i) {
#ifdef OMP_ON
        omp_set_num_threads(static_cast<int>(powf(2, i)));
#endif
        v_time[i] = magma_util::measure_duration("", false, [&]() -> void {
            exa::minmax_element(v_input, 0, v_input.size(),
            [] __device__ (auto const v1, auto const v2) -> bool {
                return v1 < v2;
            });
        });
    }
    print_vector("Exa mimmax_element Times: ", v_time);
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
    print_vector("Exa reduce Times: ", v_time);
}

void sort_bm(s_vec<long long> &v_input, s_vec<long long> &v_time, int const n_iter) {
    for (long long i = 0; i < n_iter; ++i) {
#ifdef OMP_ON
        omp_set_num_threads(static_cast<int>(powf(2, i)));
#endif
        auto v_copy = v_input;
        v_time[i] = magma_util::measure_duration("", false, [&]() -> void {
            exa::sort(v_copy, 0, v_copy.size(), []__device__(auto const &v1, auto const &v2) -> bool {
                return v1 < v2;
            });
        });
    }
    print_vector("Exa sort Times: ", v_time);
}

void transform_bm(s_vec<long long> &v_input, s_vec<long long> &v_time, int const n_iter) {
    for (long long i = 0; i < n_iter; ++i) {
#ifdef OMP_ON
        omp_set_num_threads(static_cast<int>(powf(2, i)));
#endif
        v_time[i] = magma_util::measure_duration("", false, [&]() -> void {
            exa::transform(v_input, v_input, 0, v_input.size(), 0,
            [] __device__ (auto const &v) -> long long {
                return -v;
            });
        });
    }
    print_vector("Exa transform Times: ", v_time);
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
            exa::unique(v_iota, v_copy, 0, v_iota.size(), 0, [=]__device__(auto const &i) -> bool {
                return v_input[i-1] != v_input[i];
            });
        });
    }
    print_vector("Exa unique Times: ", v_time);
}

int main(int argc, char **argv) {
    std::cout << "Starting Exa Benchmark Tests" << std::endl;

    print_cuda_memory_usage();

//    int const INT32 = INT32_MAX;
    int const MB100 = 100000000;
    s_vec<long long> v_input(MB100);
    int n_iter = 1;
//    long long cnt = 0;
    std::cout << "iterations: " << n_iter << std::endl;
    s_vec<long long> v_time(n_iter);

    // iota
    iota_bm(v_input, v_time, n_iter);
    std::cout << std::endl;

    // transform
    transform_bm(v_input, v_time, n_iter);
    std::cout << std::endl;

    // count_if
//    magma_util::measure_duration("std count_if: ", true, [&]() -> void {
//        cnt = std::count_if(v_input.begin(), v_input.end(), [](auto const &v) -> bool {
//            return v % 2 == 0;
//        });
//    });
//    std::cout << "count result: " << cnt << std::endl;
    count_if_bm(v_input, v_time, n_iter);
    std::cout << std::endl;

    // minmax_element
//    magma_util::measure_duration("std minmax_element: ", true, [&]() -> void {
//        std::minmax_element(v_input.begin(), v_input.end());
//    });
    minmax_element_bm(v_input, v_time, n_iter);
    std::cout << std::endl;

    // fill
//    magma_util::measure_duration("std fill: ", true, [&]() -> void {
//        std::fill(v_input.begin(), v_input.end(), 100);
//    });
    fill_bm(v_input, v_time, n_iter);
    std::cout << std::endl;

    // copy_if
    v_input.resize(MB100);
    exa::iota(v_input, 0, v_input.size(), 0);
    s_vec<long long> v_copy(v_input.size());
    copy_bm(v_input, v_copy, v_time, n_iter);
    std::cout << std::endl;

    // reduce
    reduce_bm(v_input, v_time, n_iter);
    std::cout << std::endl;

    // for_each
    v_copy.resize(100);
//    magma_util::measure_duration("std for_each: ", true, [&]() -> void {
//        std::for_each(v_input.begin(), v_input.end(), [&](auto &v) -> void {
//            long long sum = 0;
//            for (auto const &v2 : v_copy) {
//                sum += v2;
//            }
//            v = sum;
//        });
//    });
    for_each_bm(v_input, v_copy, v_time, n_iter);
    std::cout << std::endl;

    // sort
    v_input.resize(MB100);
//    random_vector(v_input, v_input.size() / 2);
    v_copy = v_input;
//    magma_util::measure_duration("std sort: ", true, [&]() -> void {
//        std::sort(v_copy.begin(), v_copy.end());
//    });
//    sort_bm(v_input, v_time, n_iter);
//    std::cout << std::endl;

    // unique
//    v_input = v_copy;
//    magma_util::measure_duration("std unique: ", true, [&]() -> void {
//        auto last = std::unique(v_copy.begin(), v_copy.end());
//        v_copy.erase(last, v_copy.end());
//    });
//    unique_bm(v_input, v_time, n_iter);

}