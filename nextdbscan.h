//
// Created by Ernir Erlingsson on 16.8.2020.
//

#ifndef NEXTDBSCAN20_NEXTDBSCAN_H
#define NEXTDBSCAN20_NEXTDBSCAN_H

#include <string>
//#include "data_process.h"
#ifdef CUDA_ON
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
template <typename T>
using h_vec = thrust::host_vector<T>;
template <typename T>
using d_vec = thrust::device_vector<T>;
#else
#include <vector>
template <typename T>
using s_vec = std::vector<T>;
template <typename T>
using h_vec = std::vector<T>;
template <typename T>
using d_vec = std::vector<T>;
#endif

#include "magma_mpi.h"

namespace nextdbscan {

    struct result {
        int clusters;
        int noise;
        int core_count;
        long long processed;
        int n;
        // TODO avoid a memory leak
//        int *point_clusters;
    };

    result start(int m, float e, int n_thread, const std::string &in_file, magmaMPI mpi) noexcept;

}


#endif //NEXTDBSCAN20_NEXTDBSCAN_H
