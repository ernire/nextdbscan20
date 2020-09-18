//
// Created by Ernir Erlingsson on 18.9.2020.
//
#include <iostream>
#include <omp.h>
#include "magma_meta.h"

void count_if_bm(s_vec<long long> &v_input, s_vec<long long> &v_time, int const n_iter) {
    for (long long i = 0; i < n_iter; ++i) {
#ifdef OMP_ON
        omp_set_num_threads(static_cast<int>(powf(2, i)));
#endif
        v_time[i] = magma_util::measure_duration("Determine Data Boundaries: ", false, [&]() -> void {
            auto cnt = exa::count_if(v_input, 0, v_input.size(), [](auto const &v) -> bool {
                return v % 2 == 0;
            });
        });
    }
    magma_util::print_vector("Exa count_if Times: ", v_time);
}

void fill_bm(s_vec<long long> &v_input, s_vec<long long> &v_time, int const n_iter) {
    for (long long i = 0; i < n_iter; ++i) {
#ifdef OMP_ON
        omp_set_num_threads(static_cast<int>(powf(2, i)));
#endif
        v_time[i] = magma_util::measure_duration("Determine Data Boundaries: ", false, [&]() -> void {
            exa::fill(v_input, 0, v_input.size(), i);
        });
    }
    magma_util::print_vector("Exa Fill Times: ", v_time);
}

void iota_bm(s_vec<long long> &v_input, s_vec<long long> &v_time, int const n_iter) {
    for (long long i = 0; i < n_iter; ++i) {
#ifdef OMP_ON
        omp_set_num_threads(static_cast<int>(powf(2, i)));
#endif
        v_time[i] = magma_util::measure_duration("Determine Data Boundaries: ", false, [&]() -> void {
            exa::iota(v_input, 0, v_input.size(), i);
        });
    }
    magma_util::print_vector("Exa iota Times: ", v_time);
}

void minmax_element_bm(s_vec<long long> &v_input, s_vec<long long> &v_time, int const n_iter) {
    for (long long i = 0; i < n_iter; ++i) {
#ifdef OMP_ON
        omp_set_num_threads(static_cast<int>(powf(2, i)));
#endif
        v_time[i] = magma_util::measure_duration("Determine Data Boundaries: ", false, [&]() -> void {
            exa::minmax_element(v_input, 0, v_input.size(), [](auto const v1, auto const v2) -> bool {
                return v1 < v2;
            });
        });
    }
    magma_util::print_vector("Exa mimmax_element Times: ", v_time);
}


int main(int argc, char **argv) {
    std::cout << "Starting Exa Benchmark Tests" << std::endl;

    int const INT32 = INT32_MAX;
    s_vec<long long> v_input(INT32);
    int n_iter = 1;
    int cnt = 0;
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

    // count_if
    magma_util::measure_duration("std count_if: ", true, [&]() -> void {
        cnt = std::count_if(v_input.begin(), v_input.end(), [](auto const &v) -> bool {
            return v % 2 == 0;
        });
    });
    std::cout << "count result: " << cnt << std::endl;
    count_if_bm(v_input, v_time, n_iter);

    // minmax_element
    magma_util::measure_duration("std minmax_element: ", true, [&]() -> void {
        std::minmax_element(v_input.begin(), v_input.end());
    });
    minmax_element_bm(v_input, v_time, n_iter);

    // fill
    magma_util::measure_duration("std fill: ", true, [&]() -> void {
        std::fill(v_input.begin(), v_input.end(), 100);
    });
    fill_bm(v_input, v_time, n_iter);



    // for_each

    // reduce

    // copy_if

    // transform

    // unique

    // sort

}