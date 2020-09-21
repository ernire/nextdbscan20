//
// Created by Ernir Erlingsson on 16.8.2020.
//

#ifndef NEXTDBSCAN20_MAGMA_META_H
#define NEXTDBSCAN20_MAGMA_META_H

//#ifdef CUDA_ON
//#include <thrust/host_vector.h>
//
//template <typename T>
//using s_vec = thrust::host_vector<T>;
//#else
//template <typename T>
//    using s_vec = std::vector<T>;
//    template <typename T>
//    using d_vec = std::vector<std::vector<T>>;
//#endif



#ifdef OMP_ON
#include <vector>
template <typename T>
using s_vec = std::vector<T>;
template <typename T>
using h_vec = std::vector<T>;
template <typename T>
using d_vec = std::vector<T>;
#include "magma_exa_omp.h"
#elif CUDA_ON
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
template <typename T>
using h_vec = thrust::host_vector<T>;
template <typename T>
using d_vec = thrust::device_vector<T>;
#include "magma_exa_cu.cuh"
#else
#include <vector>
template <typename T>
using s_vec = std::vector<T>;
#include "magma_exa.h"
#endif



#endif //NEXTDBSCAN20_MAGMA_META_H
